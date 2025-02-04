import 'package:boquilahub/src/rust/api/abstractions.dart';
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
  final AI? currentai;
  const ProcessingPage({
    super.key,
    required this.currentcolors,
    required this.currentai,
  });

  @override
  State<ProcessingPage> createState() => _ProcessingPageState();
}

class _ProcessingPageState extends State<ProcessingPage> {
  bool isfolderselected = false;
  bool isProcessing = false;
  bool analyzecomplete = false;
  bool shouldContinue = true;
  String nfoundimagestext = "";
  List<PredImg> listpredimgs = [];
  AI? currentAI;

  @override
  void initState() {
    currentAI = widget.currentai;
    super.initState();
  }

  void pause() {
    setState(() {
      shouldContinue = false;
    });
  }

  Future<bool?> askUserWhatToAnalyze() async {
    bool? result = await showDialog<bool>(
      context: context,
      builder: (context) {
        return AlertDialog(
          content: Text("¿Quieres analizar todo?"),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(true),
              child: Text("Sí"),
            ),
            TextButton(
              onPressed: () => Navigator.of(context).pop(false),
              child: Text("No, solo los datos que faltan"),
            ),
          ],
        );
      },
    );

    return result;
  }

  void analyzeW(bool bool) async {
    setState(() {
      isProcessing = true;
    });
    await analyze(bool);
    setState(() {
      analyzecomplete = true;
      isProcessing = false;
    });
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
      List<String> jpgFiles = filesInDirectory
          .where((file) => isSupportedIMG(file))
          .map((file) => file.path)
          .toList();
      List<PredImg> templist = [];
      for (String filepath in jpgFiles) {
        List<BBox> tempbbox = await readPredictionsFromFile(filepath);
        PredImg temppredimg = PredImg(filepath, tempbbox, tempbbox.isNotEmpty);
        templist.add(temppredimg);
      }
      setState(() {
        listpredimgs = templist;
        isfolderselected = true;
        analyzecomplete = false;
        shouldContinue = false;
        isProcessing = false;
        nfoundimagestext = "${listpredimgs.length} imágenes encontradas";
      });
    }
  }

  void selectFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      allowedExtensions: ['jpg', 'jpeg', "png"],
      type: FileType.custom,
    );
    if (result != null) {
      File file = File(result.files.single.path!);
      List<BBox> tempbbox = await readPredictionsFromFile(file.path);
      PredImg temppred = PredImg(file.path, tempbbox, tempbbox.isNotEmpty);
      setState(() {
        listpredimgs = [temppred];
        analyzecomplete = false;
        isfolderselected = false;
      });
    }
  }

  Future<void> analyze(bool analyzeonlyempty) async {
    setState(() {
      shouldContinue = true;
    });
    for (int i = 0; i < listpredimgs.length; i++) {
      if (!shouldContinue) break;
      if (analyzeonlyempty) {
        if (listpredimgs[i].listbbox.isNotEmpty) {
          continue;
        }
      }

      try {
        String temppath = listpredimgs[i].filePath;
        List<BBox> tempbbox = await detectBbox(filePath: temppath);
        if (!shouldContinue) break;
        setState(() {
          listpredimgs[i].listbbox = tempbbox;
          listpredimgs[i].wasprocessed = true;
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

    ButtonStyle botoncitostyle2 = ElevatedButton.styleFrom(
      foregroundColor: Colors.grey,
      backgroundColor: Colors.blueGrey,
      minimumSize: const Size(100, 45),
      padding: const EdgeInsets.symmetric(horizontal: 16),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.all(Radius.circular(10)),
      ),
    );

    TextStyle textito =
        TextStyle(color: widget.currentcolors[4], fontWeight: FontWeight.bold);

    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text("Selecciona", style: textito),
        const SizedBox(height: 20),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            ElevatedButton(
                style: botoncitostyle,
                onPressed: selectFolder,
                child: const Text("Carpeta")),
            ElevatedButton(
                style: botoncitostyle,
                onPressed: selectFile,
                child: const Text("Imagen")),
            ElevatedButton(
                style: botoncitostyle2,
                onPressed: () {},
                child: const Text("Video")),
            ElevatedButton(
                style: botoncitostyle2,
                onPressed: () {},
                child: const Text("Cámara")),
          ],
        ),
        const SizedBox(height: 10),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (listpredimgs.isNotEmpty)
              ElevatedButton(
                  onPressed: () async {
                    if (widget.currentai != null) {
                      if (isProcessing) {
                      } else {
                        if (!areBoxesEmpty(listpredimgs)) {
                          bool? checkall = await askUserWhatToAnalyze();
                          if (checkall != null) {
                            analyzeW(!checkall);
                          }
                        } else {
                          analyzeW(true);
                        }
                      }
                    } else {
                      simpleDialog(context, "Primero, elige una IA");
                    }
                  },
                  child: Row(
                    children: [
                      const Text("Analizar"),
                      if (isProcessing)
                        const SizedBox(
                            height: 15,
                            width: 15,
                            child: CircularProgressIndicator())
                    ],
                  )),
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
                                      for (PredImg predimg in listpredimgs) {
                                        await writePredImgToFile(predimg);
                                      }

                                      if (context.mounted) {
                                        simpleDialog(context, "✅ Listo");
                                      }
                                    },
                                    child: const Text("Exportar datos")),
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
                                          simpleDialog(context, "✅ Listo");
                                        }
                                      }
                                    },
                                    child: const Text(
                                        "Copiar imágenes \nsegún clasificación")),
                              ],
                            ),
                            title: const Text("Opciones")));
                  },
                  child: const Text("Exportar")),
            if (isProcessing)
              ElevatedButton(onPressed: pause, child: Icon(Icons.pause))
          ],
        ),
        if (isfolderselected) Text(nfoundimagestext),
        if (isfolderselected)
          Text("${countProcessedImages(listpredimgs)} imágenes procesadas"),
        const SizedBox(height: 10),
        SizedBox(
          height: MediaQuery.of(context).size.height * 0.58,
          width: MediaQuery.of(context).size.width * 0.8,
          child: ScrollConfiguration(
            behavior: MyCustomScrollBehavior(),
            child: ListView.builder(
              addAutomaticKeepAlives: false,
              shrinkWrap: true,
              scrollDirection: Axis.vertical,
              itemCount: listpredimgs.length,
              itemBuilder: (context, index) {
                return render(listpredimgs[index]);
              },
            ),
          ),
        ),
      ],
    );
  }
}

class MyCustomScrollBehavior extends MaterialScrollBehavior {
  // Override behavior methods and getters like dragDevices
  @override
  Set<ui.PointerDeviceKind> get dragDevices => {
        ui.PointerDeviceKind.touch,
        ui.PointerDeviceKind.mouse,
        ui.PointerDeviceKind.trackpad,
      };
}
