String cudaText(double cudaversion){
  if (cudaversion == 12.8){
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