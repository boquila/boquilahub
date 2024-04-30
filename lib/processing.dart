import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:boquilahub/src/rust/api/simple.dart';
import 'package:boquilahub/src/resources/objects.dart';
import 'package:boquilahub/src/resources/utils.dart';

class ProcessingPage extends StatefulWidget {
  final List<Color> currentcolors;
  final AI currentAI;
  const ProcessingPage(
      {super.key,
      required this.currentAI,
      required this.currentcolors,
      required this.title});

  final String title;

  @override
  State<ProcessingPage> createState() => _ProcessingPageState();
}

class _ProcessingPageState extends State<ProcessingPage> {
  bool isfolderselected = false;
  bool isfileselected = false;
  bool isProcessingFolder = false;
  bool isProcessingSingle = false;
  int nProcessed = 0;
  bool analyzecomplete = false;
  String foundImagesText = "";
  late String jpgFile;
  late List<String> jpgFiles;
  late List<BBox> animalDataList;
  late PredImg predimg;

  void selectFolder() async {
    String? selectedDirectory = await FilePicker.platform.getDirectoryPath();

    if (selectedDirectory != null) {
      final List<FileSystemEntity> entities =
          await Directory(selectedDirectory.toString()).list().toList();
      final Iterable<File> filesInDirectory = entities.whereType<File>();
      print(selectedDirectory);
      print(filesInDirectory);

      setState(() {
        jpgFiles = filesInDirectory
            .where((file) =>
                file.path.toLowerCase().endsWith('.jpg') ||
                file.path.toLowerCase().endsWith('.jpeg') ||
                file.path.toLowerCase().endsWith('.JPG') ||
                file.path.toLowerCase().endsWith('.JPEG'))
            .map((file) => file.path)
            .toList();
        nProcessed = 0;
        foundImagesText = "${jpgFiles.length} imágenes encontradas";
        isfolderselected = true;
        isfileselected = false;
      });
    } else {
      // User canceled the picker
    }
  }

  Future<bool> isImageCorrupted(String filePath) async {
    try {
      // Read the file as bytes
      Uint8List bytes = File(filePath).readAsBytesSync();
      // Attempt to decode the image
      ui.Codec codec = await ui.instantiateImageCodec(bytes);
      // Check if the image is decoded successfully
      await codec.getNextFrame();

      return false; // Image is not corrupted
    } catch (e) {
      print("Error: $e");
      return true; // Image is corrupted or incomplete
    }
  }

  void selectFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles();
    if (result != null) {
      File file = File(result.files.single.path!);
      print(file.path);

      bool isPicture = file.path.endsWith(".jpg") |
          file.path.endsWith(".JPG") |
          file.path.endsWith(".jpeg") |
          file.path.endsWith(".JPEG");
      print(isPicture);

      if (isPicture) {
        setState(() {
          jpgFile = file.path;
          // bool sd = await isImageCorrupted(filePath);
          // print("is corrupted?");
          // print(sd);
          analyzecomplete = false;
          isfolderselected = false;
          isfileselected = true;
        });
      }
    } else {
      // User canceled the picker
    }
  }

  Future<String?> analyzeSingleFile(String filePath) async {
    print("Sending to Rust");
    print(filePath);
    try {
      String response = await detect(filePath: filePath);
      return response;
    } catch (e) {
      print(e);
    }
    return null;
  }

  Future<List<String?>> analyzefolder(List<String> filePaths) async {
    print("Sending to Rust");
    List<String?> responses = [];
    for (String filePath in filePaths) {
      print(filePath);
      try {
        String response = await detect(filePath: filePath);
        responses.add(response);
      } catch (e) {
        responses.add("error");
      }
      setState(() {
        nProcessed = nProcessed + 1;
      });
    }
    return responses;
  }

  @override
  Widget build(BuildContext context) {
    ButtonStyle botoncitostyle = ElevatedButton.styleFrom(
      foregroundColor: widget.currentcolors[0],
      backgroundColor: widget.currentcolors[4],
      minimumSize: const Size(100, 45),
      padding: const EdgeInsets.symmetric(horizontal: 16),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.all(Radius.circular(10)),
      ),
    );

    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        const SizedBox(height: 20),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
                style: botoncitostyle,
                onPressed: selectFolder,
                child: const Text("Carpeta")),
            const SizedBox(width: 50),
            ElevatedButton(
                style: botoncitostyle,
                onPressed: selectFile,
                child: const Text("Imagen"))
          ],
        ),
        const SizedBox(height: 10),
        if (isfolderselected)
          ElevatedButton(
              onPressed: () async {
                setState(() {
                  isProcessingFolder = true;
                  nProcessed = 0;
                });
                List<String?> results = await analyzefolder(jpgFiles);
                setState(() {
                  isProcessingFolder = false;
                });
              },
              child: const Text("Analizar carpeta")),
        if (isfolderselected) Text(foundImagesText),
        Row(
          children: [
            if (isfolderselected) Text("$nProcessed imágenes procesadas"),
            if (isProcessingFolder) const CircularProgressIndicator(),
          ],
        ),
        if (isfileselected)
          ElevatedButton(
              onPressed: () async {
                setState(() {
                  isProcessingSingle = true;
                });
                String? results = await analyzeSingleFile(jpgFile);
                setState(() {
                  isProcessingSingle = false;
                });
                setState(() {
                  if (results != null) {
                    List<dynamic> jsonList = json.decode(results);
                    print(jsonList);
                    animalDataList =
                        jsonList.map((json) => BBox.fromJson(json,widget.currentAI)).toList();

                    predimg = PredImg(File(jpgFile), animalDataList);
                    

                    analyzecomplete = true;
                  }
                });
              },
              child: const Text("Analizar fotografía")),
        const SizedBox(height: 10),
        if (isfileselected & !analyzecomplete)
          SizedBox(
              width: MediaQuery.of(context).size.width * 0.8,
              height: MediaQuery.of(context).size.height * 0.58,
              child: Center(child: Image.file(File(jpgFile)))),
        if (isfileselected & analyzecomplete)
          SizedBox(
              width: MediaQuery.of(context).size.width * 0.8,
              height: MediaQuery.of(context).size.height * 0.58,
              child: Center(
                child: predimg.render(),
              )),
        // child: BoxImage(
        //     image: Image.file(File(filePath)),
        //     listBBox: animalDataList))),
        const SizedBox(height: 10),
        if (isfileselected & analyzecomplete)
          ElevatedButton(
              onPressed: () {
                showDialog(
                    context: context,
                    builder: (context) => AlertDialog(
                            actions: [
                              ElevatedButton(
                                  onPressed: () {
                                    Navigator.pop(context);
                                  },
                                  child: const Text("Exportar"))
                            ],
                            content: const Text("Contenido"),
                            title: const Text("Opciones")));
              },
              child: const Text("Exportar analisis")),
        const SizedBox(height: 10),
      ],
    );
  }
}
