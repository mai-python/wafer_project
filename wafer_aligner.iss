[Setup]
AppName=Wafer Aligner
AppVersion=1.0
DefaultDirName={autopf}\Wafer_Aligner
DefaultGroupName=Wafer Aligner
UninstallDisplayIcon={app}\wafer_aligner.exe
OutputDir=C:\Users\Woody_msiPC\Desktop\wafer_aligner\Output
OutputBaseFilename=Wafer_Aligner_Installer
Compression=lzma
SolidCompression=yes
DisableDirPage=no
SetupIconFile=C:\Users\Woody_msiPC\Desktop\wafer_aligner\cwu_icon.ico

[Files]
Source: "C:\Users\Woody_msiPC\Desktop\wafer_aligner\wafer_aligner.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\Woody_msiPC\Desktop\wafer_aligner\splash.png"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\Woody_msiPC\Desktop\wafer_aligner\models\*"; DestDir: "{app}\models"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "C:\Users\Woody_msiPC\Desktop\wafer_aligner\recordings\*"; DestDir: "{app}\recordings"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "C:\Users\Woody_msiPC\Desktop\wafer_aligner\log and summary\*"; DestDir: "{app}\log and summary"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Wafer Aligner"; Filename: "{app}\wafer_aligner.exe"; WorkingDir: "{app}"
Name: "{commondesktop}\Wafer Aligner"; Filename: "{app}\wafer_aligner.exe"; Tasks: desktopicon; WorkingDir: "{app}"

[Tasks]
Name: "desktopicon"; Description: "바탕화면에 아이콘 생성"; GroupDescription: "추가 아이콘:"
