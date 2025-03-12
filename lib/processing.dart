import 'package:boquilahub/src/rust/api/abstractions.dart';
import 'package:boquilahub/src/rust/api/eps.dart';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:boquilahub/src/rust/api/inference.dart';
import 'package:boquilahub/src/rust/api/exportutils.dart';
import 'package:boquilahub/src/rust/api/video_file.dart';
import 'package:boquilahub/src/rust/api/rest.dart';
import 'package:boquilahub/src/resources/objects.dart';
import 'package:boquilahub/src/resources/palettes.dart';
import 'dart:core';

class ProcessingPage extends StatefulWidget {
  final AI? currentai;
  final EP currentep;
  final String? url;
  const ProcessingPage({
    super.key,
    required this.currentai,
    required this.currentep,
    required this.url,
  });

  @override
  State<ProcessingPage> createState() => _ProcessingPageState();
}

class _ProcessingPageState extends State<ProcessingPage> {
  bool isfolderselected = false;
  bool isvideoselected = false;
  bool isProcessing = false;
  bool analyzecomplete = false;
  bool shouldContinue = true;
  bool errorocurred = false;
  String? videoFile;
  String nfoundimagestext = "";
  List<PredImg> listpredimgs = [];

  @override
  void initState() {
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

  void analyzeW(bool bool, context) async {
    setState(() {
      isProcessing = true;
    });
    if (isvideoselected && videoFile != null) {
      if (widget.currentep.local) {
        await predictVideofile(filePath: videoFile!);
      } else {
        predictVideofileRemotely(
            filePath: videoFile!, url: "${widget.url!}/upload");
      }
      if (isvideoselected && videoFile != null && context.mounted) {
        simpleDialog(context, "Video exportado con predicciones");
      }
    } else {
      await analyze(bool);
    }

    setState(() {
      analyzecomplete = true;
      isProcessing = false;
    });
  }

  void handleAnalysisRequest(context) async {
    if (widget.currentai == null && widget.currentep.local) {
      simpleDialog(context, "Primero, elige una IA");
      return;
    }

    if (isProcessing) return;

    if (areBoxesEmpty(listpredimgs)) {
      analyzeW(true, context);
    } else {
      final checkall = await askUserWhatToAnalyze();
      if (checkall != null) {
        analyzeW(!checkall, context);
      }
    }
  }

  bool isSupportedIMG(File file) {
    bool isPicture = file.path.toLowerCase().endsWith('.jpg') ||
        file.path.toLowerCase().endsWith('.png') ||
        file.path.toLowerCase().endsWith('.webp') ||
        file.path.toLowerCase().endsWith('.gif') ||
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
        isvideoselected = false;
        analyzecomplete = false;
        shouldContinue = false;
        isProcessing = false;
        nfoundimagestext = "${listpredimgs.length} imágenes encontradas";
      });
    }
  }

  void selectFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      allowedExtensions: ['jpg', 'jpeg', "png", "webp", "gif"],
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
        isvideoselected = false;
      });
    }
  }

  void selectVideoFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      allowedExtensions: <String>[
        "mp4",
        "mov",
        "avi",
        "mkv",
        "flv",
        "webm",
        "wmv",
        "mpeg"
      ],
      type: FileType.custom,
    );
    if (result != null) {
      setState(() {
        isvideoselected = true;
        videoFile = "my_file.mp4";
        analyzecomplete = false;
        isfolderselected = false;
      });
    }
  }

  Future<void> analyze(bool analyzeonlyempty) async {
    setState(() {
      shouldContinue = true;
      errorocurred = false;
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
        List<BBox> tempbbox = [];
        if (widget.currentep.local) {
          tempbbox = await detectBbox(filePath: temppath);
        } else {
          tempbbox = await detectBboxRemotely(
              url: "${widget.url!}/upload", filePath: temppath);
        }
        if (!shouldContinue) break;
        setState(() {
          listpredimgs[i].listbbox = tempbbox;
          listpredimgs[i].wasprocessed = true;
        });
      } catch (e) {
        setState(() {
          errorocurred = true;
        });
      }
    }
  }

  Widget _buildSourceButton(
      {required IconData icon,
      required String label,
      required VoidCallback onPressed,
      required Color color}) {
    return ElevatedButton.icon(
      icon: Icon(icon, size: 20),
      label: Text(label),
      onPressed: onPressed,
      style: ElevatedButton.styleFrom(
        foregroundColor: Colors.white,
        backgroundColor: color,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(10),
        ),
      ),
    );
  }

  Widget _buildDataSourceButtons() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        _buildSourceButton(
            icon: Icons.folder_open,
            label: "Carpeta",
            onPressed: selectFolder,
            color: terra[4]),
        _buildSourceButton(
            icon: Icons.image_outlined,
            label: "Imagen",
            onPressed: selectFile,
            color: terra[4]),
        _buildSourceButton(
            icon: Icons.videocam_outlined,
            label: "Video",
            onPressed: selectVideoFile,
            color: terra[4]),
        _buildSourceButton(
            icon: Icons.camera_alt_outlined,
            label: "Cámara",
            onPressed: () {},
            color: Colors.grey),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    TextStyle textito = TextStyle(color: terra[4], fontWeight: FontWeight.bold);

    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text("Selecciona", style: textito),
        const SizedBox(height: 20),
        _buildDataSourceButtons(),
        const SizedBox(height: 10),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (listpredimgs.isNotEmpty || isvideoselected)
              ElevatedButton(
                  onPressed: () async {
                    handleAnalysisRequest(context);
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
        if (errorocurred)
          Text(
            "Ha ocurrido un error en la inferencia. \nProceso cancelado.",
            textAlign: TextAlign.center,
          ),
        if (isfolderselected) Text(nfoundimagestext),
        if (isfolderselected)
          Text("${countProcessedImages(listpredimgs)} imágenes procesadas"),
        const SizedBox(height: 20),
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
