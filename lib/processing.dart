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
  bool isProcessing = false;
  bool analyzecomplete = false;
  bool shouldContinue = true;
  int nProcessed = 0;
  String nfoundimagestext = "";
  List<PredImg> listpredimgs = [];

  @override
  void initState() {
    super.initState();
  }

  bool isSupportedIMG(File file) {
    bool isPicture = file.path.toLowerCase().endsWith('.jpg') ||
        file.path.toLowerCase().endsWith('.png') ||
        file.path.toLowerCase().endsWith('.jpeg');
    return isPicture;
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

  void selectFolder() async {
    String? selectedDirectory = await FilePicker.platform.getDirectoryPath();
    if (selectedDirectory != null) {
      final List<FileSystemEntity> entities =
          await Directory(selectedDirectory.toString()).list().toList();
      final Iterable<File> filesInDirectory = entities.whereType<File>();
      setState(() {
        List<String> jpgFiles = filesInDirectory
            .where((file) => isSupportedIMG(file))
            .map((file) => file.path)
            .toList();
        listpredimgs = jpgFiles.map((file) => PredImg(file, [])).toList();
        nProcessed = 0;
        nfoundimagestext = "${listpredimgs.length} imágenes encontradas";
        isfolderselected = true;
        analyzecomplete = false;
        shouldContinue = false;
      });
    }
  }

  void selectFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles();
    if (result != null) {
      File file = File(result.files.single.path!);
      if (isSupportedIMG(file)) {
        setState(() {
          PredImg temppred = PredImg(file.path, []);
          listpredimgs.add(temppred);
          analyzecomplete = false;
          isfolderselected = false;
        });
      }
    }
  }

  void analyze() async {
    setState(() {
      shouldContinue = true;
    });
    for (int i = 0; i < listpredimgs.length; i++) {
      if (!shouldContinue) break;
      try {
        String temppath = listpredimgs[i].filePath;
        List<BBox> tempbbox = await detectBbox(filePath: temppath);
        setState(() {
          listpredimgs[i] = PredImg(temppath, tempbbox);
          nProcessed = nProcessed + 1;
        });
        // ignore: empty_catches
      } catch (e) {}
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
            if (listpredimgs.isNotEmpty)
              ElevatedButton(
                  onPressed: () async {
                    if (isProcessing) {
                    } else {
                      setState(() {
                        isProcessing = true;
                        nProcessed = 0;
                      });
                      analyze();
                      setState(() {
                        analyzecomplete = true;
                        isProcessing = false;
                      });
                    }
                  },
                  child: const Text("Analizar")),
            // AI predicitons exports are done here
            if (analyzecomplete)
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
            if (isProcessing)
              const SizedBox(
                  height: 15, width: 15, child: CircularProgressIndicator())
          ],
        ),
        if (isfolderselected) Text(nfoundimagestext),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (isfolderselected) Text("$nProcessed imágenes procesadas"),
            if (isProcessing)
              const SizedBox(
                  height: 15, width: 15, child: CircularProgressIndicator()),
          ],
        ),
        if (true) const SizedBox(height: 10),
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
