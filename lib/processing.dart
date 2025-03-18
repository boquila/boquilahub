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

class MediaState {
  bool isFolderSelected = false;
  bool isVideoSelected = false;
  bool isProcessing = false;
  bool isAnalyzeComplete = false;
  bool shouldContinue = true;
  bool hasError = false;
}

class _ProcessingPageState extends State<ProcessingPage> {
  MediaState state = MediaState();
  String? videoFile;
  String nfoundimagestext = "";
  List<PredImg> listpredimgs = [];

  @override
  void initState() {
    super.initState();
  }

  void pause() {
    setState(() {
      state.shouldContinue = false;
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
      state.isProcessing = true;
    });
    if (state.isVideoSelected && videoFile != null) {
      if (widget.currentep.local) {
        await predictVideofile(filePath: videoFile!, n: BigInt.from(3));
      } else {
        predictVideofileRemotely(
            filePath: videoFile!,
            url: "${widget.url!}/upload",
            n: BigInt.from(3));
      }
      if (state.isVideoSelected && videoFile != null && context.mounted) {
        simpleDialog(context, "Video exportado con predicciones");
      }
    } else {
      await analyze(bool);
    }

    setState(() {
      state.isVideoSelected = true;
      state.isProcessing = false;
    });
  }

  void handleAnalysisRequest(context) async {
    if (widget.currentai == null && widget.currentep.local) {
      simpleDialog(context, "Primero, elige una IA");
      return;
    }

    if (state.isProcessing) return;

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
        List<BBox> tempbbox =
            await readPredictionsFromFile(inputPath: filepath);
        PredImg temppredimg = PredImg(filepath, tempbbox, tempbbox.isNotEmpty);
        templist.add(temppredimg);
      }
      setState(() {
        listpredimgs = templist;
        state.isFolderSelected = true;
        state.isVideoSelected= false;
        state.isAnalyzeComplete = false;
        state.shouldContinue = false;
        state.isProcessing = false;
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
      List<BBox> tempbbox = await readPredictionsFromFile(inputPath: file.path);
      PredImg temppred = PredImg(file.path, tempbbox, tempbbox.isNotEmpty);
      setState(() {
        listpredimgs = [temppred];
        state.isAnalyzeComplete = false;
        state.isFolderSelected = false;
        state.isVideoSelected = false;
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
      print(result.files.single.path);
      setState(() {
        state.isVideoSelected = true;
        videoFile = result.files.single.path;
        state.isAnalyzeComplete = false;
        state.isFolderSelected = false;
      });
    }
  }

  Future<void> analyze(bool analyzeonlyempty) async {
    setState(() {
      state.shouldContinue = true;
      state.hasError = false;
    });
    for (int i = 0; i < listpredimgs.length; i++) {
      if (!state.shouldContinue) break;
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
        if (!state.shouldContinue) break;
        setState(() {
          listpredimgs[i].listbbox = tempbbox;
          listpredimgs[i].wasprocessed = true;
        });
      } catch (e) {
        setState(() {
          state.hasError = true;
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
            if (listpredimgs.isNotEmpty || state.isVideoSelected)
              ElevatedButton(
                  onPressed: () async {
                    handleAnalysisRequest(context);
                  },
                  child: Row(
                    children: [
                      const Text("Analizar"),
                      if (state.isProcessing)
                        const SizedBox(
                            height: 15,
                            width: 15,
                            child: CircularProgressIndicator())
                    ],
                  )),
            if (state.isAnalyzeComplete && !state.isVideoSelected)
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
                                        ImgPred temp = ImgPred(
                                            filePath: predimg.filePath,
                                            listBbox: predimg.listbbox,
                                            wasprocessed: true);
                                        await writePredImgToFile(predImg: temp);
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
            if (state.isProcessing)
              ElevatedButton(onPressed: pause, child: Icon(Icons.pause))
          ],
        ),
        if (state.hasError)
          Text(
            "Ha ocurrido un error en la inferencia. \nProceso cancelado.",
            textAlign: TextAlign.center,
          ),
        if (state.isFolderSelected) Text(nfoundimagestext),
        if (state.isFolderSelected)
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
