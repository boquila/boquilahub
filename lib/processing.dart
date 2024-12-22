import 'package:intl/intl.dart';
import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';

import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:boquilahub/src/rust/api/inference.dart';
import 'package:boquilahub/src/resources/objects.dart';

class ProcessingPage extends StatefulWidget {
  final List<Color> currentcolors;
  final AI currentAI;
  const ProcessingPage({
    super.key,
    required this.currentcolors,
    required this.currentAI,
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
  String foundImagesText = "";
  List<String> jpgFiles = [];
  List<PredImg> listpredimgs = [];

  void selectFolder() async {
    String? selectedDirectory = await FilePicker.platform.getDirectoryPath();
    if (selectedDirectory != null) {
      final List<FileSystemEntity> entities =
          await Directory(selectedDirectory.toString()).list().toList();
      final Iterable<File> filesInDirectory = entities.whereType<File>();
      // print(selectedDirectory);
      // print(filesInDirectory);

      setState(() {
        jpgFiles = filesInDirectory
            .where((file) =>
                file.path.toLowerCase().endsWith('.jpg') ||
                file.path.toLowerCase().endsWith('.png') ||
                file.path.toLowerCase().endsWith('.PNG') ||
                file.path.toLowerCase().endsWith('.jpeg') ||
                file.path.toLowerCase().endsWith('.JPG') ||
                file.path.toLowerCase().endsWith('.JPEG'))
            .map((file) => file.path)
            .toList();
        nProcessed = 0;
        foundImagesText = "${jpgFiles.length} imágenes encontradas";
        isfolderselected = true;
        isfileselected = false;
        analyzecomplete = false;
        shouldContinue = false;
        listpredimgs = [];
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
      // print("Error: $e");
      return true; // Image is corrupted or incomplete
    }
  }

  void selectFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles();
    if (result != null) {
      File file = File(result.files.single.path!);
      // print(file.path);

      bool isPicture = file.path.endsWith(".jpg") |
          file.path.endsWith(".JPG") |
          file.path.endsWith(".jpeg") |
          file.path.endsWith(".JPEG");
      // print(isPicture);

      if (isPicture) {
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
    } else {
      // User canceled the picker
    }
  }

  Future<String?> analyzeSingleFile(String filePath) async {
    // print("Sending to Rust");
    // print(filePath);
    setState(() {
      isrunning = true;
    });
    try {
      String response = await detect(filePath: filePath);
      setState(() {
        isrunning = false;
      });
      return response;
    } catch (e) {
      print(e);
    }
    setState(() {
      isrunning = false;
    });
    return null;
  }

  Future<List<String?>> analyzefolder(List<String> filePaths) async {
    setState(() {
      shouldContinue = true;
    });
    // print("Sending to Rust");
    List<String?> responses = [];
    for (String filePath in filePaths) {
      // print(filePath);
      if (!shouldContinue) break;
      try {
        String response = await detect(filePath: filePath);
        responses.add(response);
        List<dynamic> jsonList = json.decode(response);
        List<BBox> bboxpreds = listdynamictoBBOX(jsonList, widget.currentAI);
        setState(() {
          listpredimgs.add(PredImg(filePath, bboxpreds));
        });
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
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (isfileselected)
              ElevatedButton(
                  onPressed: () async {
                    setState(() {
                      isProcessingSingle = true;
                    });
                    String? results = await analyzeSingleFile(jpgFiles[0]);
                    setState(() {
                      isProcessingSingle = false;
                    });
                    setState(() {
                      if (results != null) {
                        List<dynamic> jsonList = json.decode(results);
                        List<BBox> bboxpreds =
                            listdynamictoBBOX(jsonList, widget.currentAI);

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
                      List<String?> results = await analyzefolder(jpgFiles);
                      // bool check = results.length == jpgFiles.length;
                      // if (check) {
                      //   // List<List<dynamic>> totaljsonList = [];
                      //   // for (String? result in results) {
                      //   //   List<dynamic> jsonList = json.decode(result!);
                      //   //   totaljsonList.add(jsonList);
                      //   // }

                      //   // List<List<BBox>> totallistbbox = [];
                      //   // for (List<dynamic> jsonList in totaljsonList) {
                      //   //   List<BBox> bboxpreds =
                      //   //       listdynamictoBBOX(jsonList, widget.currentAI);
                      //   //   totallistbbox.add(bboxpreds);
                      //   // }

                      //   // List<PredImg> templistpredimgs = [];
                      //   // for (int i = 0; i < totallistbbox.length; i++) {
                      //   //   PredImg temppredimg =
                      //   //       PredImg(jpgFiles[i], totallistbbox[i]);
                      //   //   templistpredimgs.add(temppredimg);
                      //   // }
                      //   // setState(() {
                      //   //   listpredimgs = templistpredimgs;
                      //   // });
                      // } else {
                      //   if (!context.mounted) return;
                      //   niceError(context);
                      // }
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
                                      writeCsv(
                                          listpredimgs, "analisis_$str.csv");
                                      writeCsv2(listpredimgs,
                                          "analisis_condensado_$str.csv");
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
                                        processFinishedCheckMark(context);
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
            if (isrunning) const SizedBox(height: 15, width: 15, child: CircularProgressIndicator())
          ],
        ),
        if (isfolderselected) Text(foundImagesText),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (isfolderselected) Text("$nProcessed imágenes procesadas"),
            if (isProcessingFolder) const SizedBox(height: 15, width: 15, child: CircularProgressIndicator()),
          ],
        ),
        const SizedBox(height: 10),
        // if (isfileselected & !analyzecomplete)
        //   SizedBox(
        //       width: MediaQuery.of(context).size.width * 0.8,
        //       height: MediaQuery.of(context).size.height * 0.58,
        //       child: Center(child: Image.file(File(jpgFiles[0])))),
        // SizedBox(
        //   height: MediaQuery.of(context).size.height * 0.58,
        //   width: MediaQuery.of(context).size.width * 0.8,
        //   child: ListView(
        //     shrinkWrap: true,
        //     scrollDirection: Axis.horizontal,
        //     children: <Widget>[
        //       for (String jpgFile in jpgFiles)
        //         Center(child: Image.file(File(jpgFile)))
        //     ],
        //   ),
        // ),
        SizedBox(
          height: MediaQuery.of(context).size.height * 0.58,
          width: MediaQuery.of(context).size.width * 0.8,
          child: ListView(
            shrinkWrap: true,
            scrollDirection: Axis.vertical,
            children: <Widget>[
              for (PredImg predimg in listpredimgs) predimg.render(),
            ],
          ),
        ),
        // SizedBox(
        //   height: MediaQuery.of(context).size.height * 0.58,
        //   width: MediaQuery.of(context).size.width * 0.8,
        //   child: ListView(
        //     // This next line does the trick.
        //     scrollDirection: Axis.horizontal,
        //     children: <Widget>[
        //       for (PredImg predimg in listpredimgs) showpredimg(predimg, context),
        //     ],
        //   ),
        // ),
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

List<BBox> listdynamictoBBOX(List<dynamic> jsonList, AI ai) {
  List<BBox> bboxpreds =
      jsonList.map((json) => BBox.fromJson(json, ai)).toList();
  return bboxpreds;
}

niceError(context) {
  return showDialog(
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

Widget showpredimg(PredImg predimg, context) {
  return SizedBox(
    width: MediaQuery.of(context).size.width * 0.4,
    height: MediaQuery.of(context).size.height * 0.29,
    child: Center(
      child: predimg.render(),
    ),
  );
}
