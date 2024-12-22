import 'dart:io';
// This file helps us manage and understand the hardware and the dependencies that the user has


// Will get CUDA version
// if error, will return 0
Future<double> getCudaVersion() async {
  try {
    // Run the 'nvcc --version' command
    ProcessResult result = await Process.run('nvcc', ['--version']);
    
    // Check if the command was successful
    if (result.exitCode != 0) {
      return 0;
      // throw 'Command failed with exit code ${result.exitCode}';
    }
    
    // Extract the output
    String output = result.stdout.toString();
    
    // Use a regular expression to find the CUDA version
    RegExp versionRegExp = RegExp(r'release (\d+\.\d+),');
    Match? match = versionRegExp.firstMatch(output);
    
    if (match != null) {
      String versionString = match.group(1)!;
      // Convert the version string to a double
      return double.parse(versionString);
    } else {
      return 0;
    }
  } catch (e) {
    // Handle any errors
    return 0;
  }
}

Future<bool> doescuDNNexists() async {
  // Define the path to the file
  const String filePath = r'C:\Program Files\NVIDIA\CUDNN\v8.9.2.26\bin\cudnn64_8.dll';
  try {    
    return File(filePath).existsSync();
  } catch (e) {
    return false;
  }
}

String cudaText(double cudaversion){
  if (cudaversion == 12.4){
    return "se encontró la versión correcta";
  } else if (cudaversion == 0){
    return "no se encontró una versión";
  } else {
    String text = cudaversion.toString();
    return "se encontró versión $text";
  }
}

String cuDNNText(bool iscudnnAvailable){
  if (iscudnnAvailable == true){
    return "se encontró la versión correcta";
  } else {
    return "no se encontró la versión correcta";
  }
}


void addDirectoryToPath(String directory) {
  final environment = Platform.environment;
  final currentPath = environment['PATH'] ?? '';

  // Normalize directory path for comparison
  final normalizedDirectory = directory.replaceAll('\\', '/').toLowerCase();
  final normalizedPaths = currentPath.split(';').map((p) => p.replaceAll('\\', '/').toLowerCase());

  // Check if the directory is already in the PATH
  if (!normalizedPaths.contains(normalizedDirectory)) {
    print('Adding directory to PATH: $directory');

    // Use PowerShell to modify the PATH
    final powershellCommand = '''
    \$currentPath = [System.Environment]::GetEnvironmentVariable('Path', [System.EnvironmentVariableTarget]::User)
    if (-not \$currentPath.Contains("$directory")) {
        \$newPath = "\$currentPath;$directory"
        [System.Environment]::SetEnvironmentVariable('Path', \$newPath, [System.EnvironmentVariableTarget]::User)
    }
    ''';

    final powershellFilePath = 'add_to_path.ps1';
    final powershellFile = File(powershellFilePath);

    // Write the PowerShell command to a file
    powershellFile.writeAsStringSync(powershellCommand);

    // Execute the PowerShell file to update the PATH
    final result = Process.runSync(
      'powershell',
      ['-ExecutionPolicy', 'Bypass', '-File', powershellFilePath],
      runInShell: true,
    );

    if (result.exitCode == 0) {
      print('Directory added to PATH successfully.');
    } else {
      print('Failed to add directory to PATH.');
      print('Error: ${result.stderr}');
    }

    // Clean up by deleting the PowerShell file
    powershellFile.deleteSync();
  } else {
    print('Directory is already in PATH.');
  }
}