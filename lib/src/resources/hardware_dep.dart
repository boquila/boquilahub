import 'dart:io';
// This file helps us manage and understand the hardware and the dependencies that the user has

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