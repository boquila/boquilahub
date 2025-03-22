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
import 'package:boquilahub/src/rust/api/feed.dart';
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
  bool isProcessing = false;
  bool isAnalysisComplete = false;
  bool shouldContinue = true;
  bool hasError = false;

  bool imgMode = false;
  bool videoMode = false;
  bool feedMode = false;

  void setMode({bool img = false, bool video = false, bool feed = false}) {
    imgMode = img;
    videoMode = video;
    feedMode = feed;
  }
}

class _ProcessingPageState extends State<ProcessingPage> {
  MediaState state = MediaState();
  List<PredImg> listpredimgs = [];
  int? stepFrame; // for videofile and feed

  // Feed global variables
  Image? feedFramebuffer;
  Image? previousFeedFramebuffer;
  String? rtspURL;

  // Video global variables
  Image? framebuffer;
  Image? previousFramebuffer;
  int? totalFrames;
  int? currentFrame;
  String? videoFile;

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

  Future<int?> askUserForInt() async {
    int? result = await showDialog<int>(
      context: context,
      builder: (context) {
        TextEditingController controller = TextEditingController();

        return AlertDialog(
          title: Text("¿Cada cuántos frames quieres analizar?"),
          content: TextField(
            controller: controller,
            keyboardType: TextInputType.number,
            decoration: InputDecoration(
              labelText: "Elije un número entre 1 y 30",
            ),
          ),
          actions: [
            TextButton(
              onPressed: () {
                int? number = int.tryParse(controller.text);
                if (number != null && number >= 1 && number <= 30) {
                  Navigator.of(context).pop(number);
                } else {
                  // Show an error message or handle invalid input
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                        content: Text(
                            "Por favor ingresa un número valido entre 1 y 30.")),
                  );
                }
              },
              child: Text("OK"),
            ),
            TextButton(
              onPressed: () => Navigator.of(context).pop(null),
              child: Text("Cancelar"),
            ),
          ],
        );
      },
    );

    return result;
  }

  // SECTION: Analysis
  // The user has selcted some data, and now he pressed to analyze it
  void analyzeImg(BuildContext context) async {
    aiCheck(context);
    if (state.isProcessing) return; // Won't analyze

    bool checkall = true;
    if (!areBoxesEmpty(listpredimgs)) {
      final bool? response = await askUserWhatToAnalyze();
      if (response == null) return;
      checkall = !response;
    }

    processingStart();
    for (int i = 0; i < listpredimgs.length; i++) {
      if (checkall) {
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
        setState(() {
          listpredimgs[i].listbbox = tempbbox;
          listpredimgs[i].wasprocessed = true;
        });
      } catch (e) {
        setState(() {
          state.hasError = true;
        });
      }
      if (!state.shouldContinue) break;
    }
    processingEnd();
  }

  void analyzeVideoFile(context) async {
    aiCheck(context);

    int? i = await askUserForInt();
    if (i == null) return;
    stepFrame = i;

    processingStart();
    if (state.videoMode && videoFile != null) {
      final a = VideofileProcessor(filePath: videoFile!);
      final int n = (await a.getNFrames()).toInt();
      setState(() {
        totalFrames = n;
      });
      List<BBox>? tempbbox;
      if (widget.currentep.local) {
        for (int i = currentFrame!; i < n; i++) {
          if (i % stepFrame! == 0) {
            var (r, b) = await a.runExp();
            tempbbox = b;
            setState(() {
              previousFramebuffer = framebuffer;
              framebuffer = Image.memory(r);
              currentFrame = i;
            });
          } else {
            await a.runExp(vec: tempbbox);
          }
        }
      } else {
        if (i % stepFrame! == 0) {
          var (r, b) = await a.runRemotelyExp(url: "${widget.url!}/upload");
          tempbbox = b;
          setState(() {
            previousFramebuffer = framebuffer;
            framebuffer = Image.memory(r);
            currentFrame = i;
          });
        } else {
          await a.runRemotelyExp(url: "${widget.url!}/upload", vec: tempbbox);
        }
      }
      if (context.mounted) {
        simpleDialog(context, "Video exportado con predicciones");
      }
    }
    processingEnd();
  }

  void analyzeFeed(context) async {
    aiCheck(context);
    if (state.isProcessing) return; // Won't analyze
    if (rtspURL == null) return;
    processingStart();

    final a = RtspFrameIterator(url: rtspURL!);
    int i = 0;
    while (state.shouldContinue) {
      if (i % stepFrame! == 0) {
        Uint8List r;
        // ignore: unused_local_variable
        List<BBox> b;
        if (widget.currentep.local) {
          (r, b) = await a.runExp();
        } else {
          (r, b) = await a.runRemotelyExp(url: "${widget.url!}/upload");
        }
        setState(() {
          previousFeedFramebuffer = feedFramebuffer;
          feedFramebuffer = Image.memory(r);
        });
      } else {
        await a.ignoreFrame();
      }
      i = i + 1;
    }
    processingEnd();
  }

  // SECTION: Checks and validations
  void aiCheck(context) {
    if (widget.currentai == null && widget.currentep.local) {
      simpleDialog(context, "Primero, elige una IA");
      return;
    }
  }

  bool isSupportedIMG(File file) {
    bool isPicture = file.path.toLowerCase().endsWith('.jpg') ||
        file.path.toLowerCase().endsWith('.png') ||
        file.path.toLowerCase().endsWith('.webp') ||
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

  // SECTION: Select data
  // The user clicked a button to process something
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
      imgModeInitState(templist);
    }
  }

  void selectFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      allowedExtensions: ['jpg', 'jpeg', "png", "webp"],
      type: FileType.custom,
    );
    if (result != null) {
      File file = File(result.files.single.path!);
      List<BBox> tempbbox = await readPredictionsFromFile(inputPath: file.path);
      PredImg temppred = PredImg(file.path, tempbbox, tempbbox.isNotEmpty);
      imgModeInitState([temppred]);
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
        state.setMode(video: true);
        videoFile = result.files.single.path;
        baseInitState();
        currentFrame = 0;
      });
    }
  }

  void selectFeed() async {
    String? result = await showDialog<String>(
      context: context,
      builder: (context) {
        TextEditingController controller = TextEditingController();

        return AlertDialog(
          title: Text("Ingresa la URL"),
          content: TextField(
            controller: controller,
            keyboardType: TextInputType.url,
            decoration: InputDecoration(
              labelText:
                  "URL (ejemplo: rtsp://usuario:contraseña@ip:puerto/stream)",
              hintText: "rtsp://...",
            ),
          ),
          actions: [
            TextButton(
              onPressed: () {
                String url = controller.text.trim();
                if (url.isNotEmpty &&
                    (url.startsWith("rtsp://") ||
                        url.startsWith("http://") ||
                        url.startsWith("https://"))) {
                  Navigator.of(context).pop(url);
                } else {
                  // Show an error message for invalid input
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                      content: Text("Ingresa una URL válida"),
                    ),
                  );
                }
              },
              child: Text("OK"),
            ),
            TextButton(
              onPressed: () => Navigator.of(context).pop(null),
              child: Text("Cancelar"),
            ),
          ],
        );
      },
    );
    if (result != null) {
      setState(() {
        rtspURL = result;
        state.setMode(feed: true);
        baseInitState();
      });
    }
  }

  // SECTION: State sugar code
  void processingStart() {
    setState(() {
      state.shouldContinue = true;
      state.isProcessing = true;
      state.hasError = false;
    });
  }

  void processingEnd() {
    setState(() {
      state.isProcessing = false;
    });
  }

  void baseInitState() {
    setState(() {
      state.isAnalysisComplete = false;
      state.shouldContinue = false;
      state.isProcessing = false;
    });
  }

  void imgModeInitState(List<PredImg> foundImgs) {
    setState(() {
      listpredimgs = foundImgs;
      state.setMode(img: true);
      baseInitState();
    });
  }

  // SECTION: Widgets sugar code
  Widget analyzeButton(context, void Function(BuildContext context) onAnalyze) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        ElevatedButton(
          onPressed: () {
            onAnalyze(context);
          },
          child: const Text("Analizar"),
        ),
        processingIndicator(),
        pauseButton(),
      ],
    );
  }

  Widget _buildSourceButton(
      {required IconData icon,
      required String label,
      required VoidCallback onPressed}) {
    return ElevatedButton.icon(
      icon: Icon(icon, size: 20),
      label: Text(label),
      onPressed: onPressed,
      style: ElevatedButton.styleFrom(
        foregroundColor: Colors.white,
        backgroundColor: terra[4],
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
            icon: Icons.folder_open, label: "Carpeta", onPressed: selectFolder),
        _buildSourceButton(
            icon: Icons.image_outlined, label: "Imagen", onPressed: selectFile),
        _buildSourceButton(
            icon: Icons.videocam_outlined,
            label: "Video",
            onPressed: selectVideoFile),
        _buildSourceButton(
            icon: Icons.camera_alt_outlined,
            label: "Cámara",
            onPressed: selectFeed)
      ],
    );
  }

  Widget errorText() {
    if (state.hasError) {
      return Text(
        "Ha ocurrido un error en la inferencia. \nProceso cancelado.",
        textAlign: TextAlign.center,
      );
    }
    return SizedBox.shrink();
  }

  Widget processingIndicator() {
    if (state.isProcessing) {
      return const SizedBox(
          height: 15, width: 15, child: CircularProgressIndicator());
    }
    return SizedBox.shrink();
  }

  Widget pauseButton() {
    if (state.isProcessing) {
      return ElevatedButton(onPressed: pause, child: Icon(Icons.pause));
    }
    return SizedBox.shrink();
  }

  void exportImgData(context) async {
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
  }

  void copyImgs(context) async {
    String? selectedDirectory = await FilePicker.platform.getDirectoryPath();
    if (selectedDirectory != null) {
      await copyToFolder(listpredimgs, "$selectedDirectory/export");
      if (context.mounted) {
        simpleDialog(context, "✅ Listo");
      }
    }
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
        // SECTION: IMAGES (or a folder full of images)
        if (state.imgMode) ...[
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              analyzeButton(context, analyzeImg),
              if (state.isAnalysisComplete)
                ElevatedButton(
                    onPressed: () {
                      showDialog(
                          context: context,
                          builder: (context) => AlertDialog(
                              content: Row(
                                children: [
                                  ElevatedButton(
                                      onPressed: () async {
                                        exportImgData(context);
                                      },
                                      child:
                                          const Text("Exportar observaciones")),
                                  const SizedBox(width: 10),
                                  ElevatedButton(
                                      onPressed: () async {
                                        copyImgs(context);
                                      },
                                      child: const Text(
                                          "Copiar imágenes \nsegún clasificación")),
                                ],
                              ),
                              title: const Text("Opciones")));
                    },
                    child: const Text("Exportar")),
            ],
          ),
          Text("${listpredimgs.length} imágenes encontradas"),
          Text("${countProcessedImages(listpredimgs)} imágenes procesadas"),
          const SizedBox(height: 20),
          errorText(),
          displayImg(
              ScrollConfiguration(
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
              context),
        ],
        // SECTION: VIDEO FILE
        if (state.videoMode) ...[
          analyzeButton(context, analyzeVideoFile),
          if (currentFrame != null)
            Text(
                "${currentFrame! + 1} frames analizados de un total de $totalFrames"),
          if (framebuffer != null && previousFramebuffer != null)
            Stack(
              children: [
                displayImg(framebuffer!, context),
                displayImg(previousFramebuffer!, context),
              ],
            )
        ],
        // SECTION: RTSP
        if (state.feedMode) ...[
          analyzeButton(context, analyzeFeed),
          if (feedFramebuffer != null && previousFeedFramebuffer != null)
            Stack(
              children: [
                displayImg(feedFramebuffer!, context),
                displayImg(previousFeedFramebuffer!, context),
              ],
            )
        ]
      ],
    );
  }
}

Widget displayImg(Widget child, BuildContext context) {
  return SizedBox(
    height: MediaQuery.of(context).size.height * 0.58,
    width: MediaQuery.of(context).size.width * 0.8,
    child: ClipRect(
      child: child,
    ),
  );
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
