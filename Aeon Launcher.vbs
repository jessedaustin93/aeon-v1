Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

root = fso.GetParentFolderName(WScript.ScriptFullName)
scriptPath = root & "\scripts\aeon_launcher.py"
pythonw = root & "\.venv\Scripts\pythonw.exe"

shell.CurrentDirectory = root

If fso.FileExists(pythonw) Then
  shell.Run """" & pythonw & """ """ & scriptPath & """", 0, False
Else
  shell.Run "pythonw """ & scriptPath & """", 0, False
End If
