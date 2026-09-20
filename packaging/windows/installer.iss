; Inno Setup Script for JEPA Runtime
; Generates Windows installer executable with PATH registration, desktop shortcuts,
; and optional Windows service setup.

#define MyAppName "JEPA Runtime"
#define MyAppVersion "0.2.0"
#define MyAppPublisher "JEPA Engineering Team"
#define MyAppURL "https://github.com/facebookresearch/jepa"
#define MyAppExeName "jepa.exe"

[Setup]
AppId={{D37F8A6B-9B41-45A0-9A92-6F8E80C9C25F}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\JEPA
DisableProgramGroupPage=yes
LicenseFile=..\..\LICENSE
OutputDir=..\..\target
OutputBaseFilename=jepa-setup-x86_64
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ChangesEnvironment=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "envPath"; Description: "Add JEPA installation folder to system PATH"; GroupDescription: "System Integration:"
Name: "winService"; Description: "Register JEPA as a background Windows Service (port 11435)"; GroupDescription: "System Integration:"; Flags: unchecked

[Files]
Source: "..\..\target\release\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\README.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Parameters: "app"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Parameters: "app"; Tasks: desktopicon

[Registry]
; Register application directory in system PATH
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; \
    ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; \
    Check: NeedsAddPath(ExpandConstant('{app}')); Tasks: envPath

[Run]
Filename: "{app}\{#MyAppExeName}"; Parameters: "app"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_LOCAL_MACHINE,
    'SYSTEM\CurrentControlSet\Control\Session Manager\Environment',
    'Path', OrigPath)
  then begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + UpperCase(Param) + ';', ';' + UpperCase(OrigPath) + ';') = 0;
end;
