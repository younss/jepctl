const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

// Exercise the shipped script with a small DOM double; never initialize hardware,
// timers, WebGL, or network requests. No test hooks are added to production code.
function harness({ mac = true, narrow = false, stored = null } = {}) {
    const nodes = new Map();
    class Node {
        constructor(id) {
            this.id = id;
            this.dataset = {};
            this.style = { display: 'none' };
            this.attributes = {};
            this.listeners = {};
            this.scrollTop = this.scrollLeft = 0;
            this.regions = [];
            this.classes = new Set();
            this.classList = {
                add: key => this.classes.add(key),
                remove: key => this.classes.delete(key),
                contains: key => this.classes.has(key),
                toggle: (key, on) => on ? this.classes.add(key) : this.classes.delete(key),
            };
        }
        addEventListener(type, callback) { (this.listeners[type] ||= []).push(callback); }
        setAttribute(key, value) { this.attributes[key] = value; }
        getAttribute(key) { return key === 'data-section' ? this.dataset.section : this.attributes[key]; }
        querySelector(selector) {
            if (selector === '.nav-text') return { textContent: this.dataset.section };
            if (selector === '.nav-item.active') return items.find(item => item.classes.has('active'));
            return null;
        }
        querySelectorAll() { return this.regions; }
        closest() { return this.editable ? this : null; }
        contains(node) { return this === node || (this.id === 'app-sidebar' && items.includes(node)); }
        focus() { document.activeElement = this; }
        click() { for (const listener of this.listeners.click || []) listener({ target: this }); }
    }
    const node = id => {
        if (!nodes.has(id)) nodes.set(id, new Node(id));
        return nodes.get(id);
    };
    const ids = ['overview', 'models', 'image-playground', 'video-stream', 'anomaly-monitor', 'gestures', 'robot', 'security', 'settings'];
    const items = ids.map(id => { const item = node(`tab-${id}`); item.dataset.section = id; return item; });
    const sections = ids.map(id => node(`section-${id}`));
    items[0].classes.add('active');
    sections[0].classes.add('active');
    const dialogs = [node('confirm-dialog'), node('shortcuts-dialog')];
    const document = node('document');
    document.body = node('body');
    document.getElementById = node;
    document.querySelectorAll = selector => ({ '.nav-item': items, '.content-section': sections, '.modal-overlay': dialogs }[selector] || []);
    const media = { matches: narrow, addEventListener(type, listener) { this.change = listener; } };
    const storage = new Map([['jepctl_sidebar_collapsed', stored]]);
    const calls = [];
    const context = vm.createContext({
        document, navigator: { platform: mac ? 'MacIntel' : 'Win32' },
        window: { matchMedia: () => media }, console,
        localStorage: { getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value) },
    });
    const source = fs.readFileSync(path.join(__dirname, '../../src/ui/app.js'), 'utf8');
    const hooks = `globalThis.navigation = { setupNavigation, switchSection, state };
        fetchModels = () => globalThis.record('models');
        fetchGesturesList = () => globalThis.record('gestures');
        fetchApiKeys = fetchAuditLog = renderIntegrationExample = () => {};
        robotEnterSection = () => globalThis.record('robot-enter');
        robotLeaveSection = () => globalThis.record('robot-leave');`;
    context.record = name => calls.push(name);
    vm.runInContext(source.replace('// Kickoff', hooks + '\n// Kickoff'), context);
    context.navigation.setupNavigation();
    const key = (value, options = {}) => {
        const event = { key: value, target: node('main-content'), metaKey: mac, ctrlKey: !mac, preventDefault() { this.prevented = true; }, ...options };
        for (const listener of document.listeners.keydown) listener(event);
        return event;
    };
    return { ...context.navigation, node, key, media, items, document, calls, storage };
}

test('restores each workspace and nested inspector scroll without disturbing robot lifecycle', () => {
    const h = harness();
    h.switchSection('robot');
    const inspector = h.node('robot-inspector');
    h.node('section-robot').regions.push(inspector);
    inspector.scrollTop = 420;
    h.node('main-content').scrollTop = 80;
    h.switchSection('settings');
    inspector.scrollTop = 0; // A hidden view may lose its browser scroll offset.
    h.switchSection('robot');
    assert.equal(inspector.scrollTop, 420);
    assert.equal(h.node('main-content').scrollTop, 80);
    assert.deepEqual(h.calls, ['robot-enter', 'robot-leave', 'robot-enter']);
    h.switchSection('robot');
    h.switchSection('unknown');
    assert.equal(h.calls.length, 3);
    assert.equal(h.state.activeSection, 'robot');
});

test('section shortcuts select the destination and preserve focus semantics on both platforms', () => {
    for (const mac of [true, false]) {
        const h = harness({ mac });
        assert.equal(h.key('6').prevented, true);
        assert.equal(h.state.activeSection, 'gestures');
        assert.equal(h.items[5].attributes['aria-selected'], 'true');
        assert.equal(h.items[5].tabIndex, 0);
        assert.equal(h.items[0].tabIndex, -1);
        assert.equal(h.document.activeElement, h.node('main-content'));
        h.key(',');
        assert.equal(h.state.activeSection, 'settings');
    }
});

test('shortcuts do not navigate while editing, in dialogs, or on key repeats', () => {
    const h = harness();
    const input = h.node('input'); input.editable = true;
    h.key('7', { target: input });
    h.key('7', { repeat: true });
    h.key('7', { altKey: true });
    h.node('confirm-dialog').style.display = 'flex';
    h.key('7');
    assert.equal(h.state.activeSection, 'overview');
    assert.deepEqual(h.calls, []);
});

test('desktop sidebar preference survives resizing without hiding narrow-window navigation', () => {
    const h = harness({ stored: 'true' });
    assert.equal(h.document.body.classes.has('sidebar-collapsed'), true);
    assert.equal(h.node('nav-toggle').attributes['aria-expanded'], 'false');
    h.media.matches = true; h.media.change();
    assert.equal(h.document.body.classes.has('sidebar-collapsed'), false);
    h.node('nav-toggle').click();
    assert.equal(h.node('app-sidebar').classes.has('is-open'), true);
    h.media.matches = false; h.media.change();
    assert.equal(h.document.body.classes.has('sidebar-collapsed'), true);
    h.key('b');
    assert.equal(h.document.body.classes.has('sidebar-collapsed'), false);
    assert.equal(h.storage.get('jepctl_sidebar_collapsed'), 'false');
});
