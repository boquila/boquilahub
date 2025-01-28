import 'package:boquilahub/src/rust/api/abstractions.dart';
import 'package:intl/intl.dart';
// import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:boquilahub/src/rust/api/inference.dart';
import 'package:boquilahub/src/rust/api/exportutils.dart';
import 'package:boquilahub/src/resources/objects.dart';
import 'dart:core';

class ProcessingPage extends StatefulWidget {
  final List<Color> currentcolors;
  const ProcessingPage({
    super.key,
    required this.currentcolors,
  });

  @override
  State<ProcessingPage> createState() => _ProcessingPageState();
}

class _ProcessingPageState extends State<ProcessingPage> {
  bool isfolderselected = false;
  bool isfileselected = false;
  bool isProcessingFolder = false;
  bool isProcessingSingle = false;
  bool analyzecomplete = false;
  bool shouldContinue = true;
  bool isrunning = false;
  int nProcessed = 0;
  String nfoundimagestext = "";
  List<String> jpgFiles = [];
  List<PredImg> listpredimgs = [];

  @override
  void initState() {
    super.initState();
  }

  bool isSupportedIMG(file) {
    bool isPicture = file.path.toLowerCase().endsWith('.jpg') ||
        file.path.toLowerCase().endsWith('.png') ||
        file.path.toLowerCase().endsWith('.jpeg');
    return isPicture;
  }

  void selectFolder() async {
    String? selectedDirectory = await FilePicker.platform.getDirectoryPath();
    if (selectedDirectory != null) {
      final List<FileSystemEntity> entities =
          await Directory(selectedDirectory.toString()).list().toList();
      final Iterable<File> filesInDirectory = entities.whereType<File>();
      setState(() {
        jpgFiles = filesInDirectory
            .where((file) => isSupportedIMG(file))
            .map((file) => file.path)
            .toList();
        nProcessed = 0;
        nfoundimagestext = "${jpgFiles.length} imágenes encontradas";
        isfolderselected = true;
        isfileselected = false;
        analyzecomplete = false;
        shouldContinue = false;
        listpredimgs = [];
      });
    } 
  }

  Future<bool> isImageCorrupted(String filePath) async {
    try {
      Uint8List bytes = File(filePath).readAsBytesSync();
      ui.Codec codec = await ui.instantiateImageCodec(bytes);
      await codec.getNextFrame();
      return false; // Image is not corrupted
    } catch (e) {
      return true; // Image is corrupted or incomplete
    }
  }

  void selectFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles();
    if (result != null) {
      File file = File(result.files.single.path!);
      if (isSupportedIMG(file)) {
        setState(() {
          jpgFiles = [file.path];
          // bool sd = await isImageCorrupted(filePath);
          // print("is corrupted?");
          // print(sd);
          analyzecomplete = false;
          isfolderselected = false;
          isfileselected = true;
        });
      }
    } 
  }

  Future<List<BBox>?> analyzeSingleFile(String filePath) async {
    setState(() {
      isrunning = true;
    });
    try {
      List<BBox> response = await detectBbox(filePath: filePath);
      setState(() {
        isrunning = false;
      });
      
      return response;
    // ignore: empty_catches
    } catch (e) {
      
    }
    setState(() {
      isrunning = false;
    });
    return null;
  }

  void analyzefolder(List<String> filePaths) async {
    setState(() {
      shouldContinue = true;
    });
    for (String filePath in filePaths) {
      // print(filePath);
      if (!shouldContinue) break;
      try {
        List<BBox> bboxpreds = await detectBbox(filePath: filePath);
        setState(() {
          listpredimgs.add(PredImg(filePath, bboxpreds));
        });
      } catch (e) {
        // print(e);
      }
      setState(() {
        nProcessed = nProcessed + 1;
      });
    }
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
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (isfileselected)
              ElevatedButton(
                  onPressed: () async {
                    setState(() {
                      isProcessingSingle = true;
                    });
                    List<BBox>? bboxpreds =
                        await analyzeSingleFile(jpgFiles[0]);
                    setState(() {
                      isProcessingSingle = false;
                    });
                    setState(() {
                      if (bboxpreds != null) {
                        listpredimgs = [PredImg(jpgFiles[0], bboxpreds)];

                        analyzecomplete = true;
                      }
                    });
                  },
                  child: const Text("Analizar fotografía")),
            if (isfolderselected & jpgFiles.isNotEmpty)
              ElevatedButton(
                  onPressed: () async {
                    if (isProcessingFolder) {
                    } else {
                      setState(() {
                        isProcessingFolder = true;
                        nProcessed = 0;
                        listpredimgs = [];
                      });
                      analyzefolder(jpgFiles);
                      setState(() {
                        analyzecomplete = true;
                        isProcessingFolder = false;
                      });
                    }
                  },
                  child: const Text("Analizar carpeta")),
            // AI predicitons exports are done here
            if ((isfileselected || isfolderselected) & analyzecomplete)
              ElevatedButton(
                  onPressed: () {
                    showDialog(
                        context: context,
                        builder: (context) => AlertDialog(
                            content: Row(
                              children: [
                                ElevatedButton(
                                    onPressed: () async {
                                      String str =
                                          DateFormat("yyyy-MM-dd HH mm ss")
                                              .format(DateTime.now());
                                      // writeCsv(predImgs: listpredimgs, outputPath: "analisis_$str.csv");
                                      // writeCsv2(predImgs: listpredimgs, outputPath: "analisis_condensado_$str.csv");
                                      processFinishedCheckMark(context);
                                      // Navigator.pop(context);
                                    },
                                    child: const Text("Exportar CSV")),
                                const SizedBox(width: 10),
                                ElevatedButton(
                                    onPressed: () async {
                                      String? selectedDirectory =
                                          await FilePicker.platform
                                              .getDirectoryPath();
                                      if (selectedDirectory != null) {
                                        await copyToFolder(listpredimgs,
                                            "$selectedDirectory/export");
                                        if (context.mounted) {
                                          processFinishedCheckMark(context);
                                        }
                                      }

                                      // Navigator.pop(context);
                                    },
                                    child: const Text(
                                        "Copiar imágenes \nsegún clasificación")),
                              ],
                            ),
                            title: const Text("Opciones")));
                  },
                  child: const Text("Exportar")),
            if (isrunning)
              const SizedBox(
                  height: 15, width: 15, child: CircularProgressIndicator())
          ],
        ),
        if (isfolderselected) Text(nfoundimagestext),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (isfolderselected) Text("$nProcessed imágenes procesadas"),
            if (isProcessingFolder)
              const SizedBox(
                  height: 15, width: 15, child: CircularProgressIndicator()),
          ],
        ),
        const SizedBox(height: 10),
        SizedBox(
          height: MediaQuery.of(context).size.height * 0.58,
          width: MediaQuery.of(context).size.width * 0.8,
          child: ListView(
            shrinkWrap: true,
            scrollDirection: Axis.vertical,
            children: <Widget>[
              for (PredImg predimg in listpredimgs) render(predimg),
            ],
          ),
        ),
      ],
    );
  }
}

processFinishedCheckMark(context) {
  return showDialog(
    context: context,
    builder: (context) => AlertDialog(
      actions: [
        ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
              Navigator.pop(context);
            },
            child: const Text("Ok"))
      ],
      title: const Text("✅ Listo"),
    ),
  );
}

niceError(context) {
  showDialog(
    context: context,
    builder: (context) => AlertDialog(
      actions: [
        ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
            },
            child: const Text("Ok"))
      ],
      title: const Text("Hubo un error"),
    ),
  );
}
