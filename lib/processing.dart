import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'src/resources/palettes.dart';
import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:boquilahub/src/rust/api/simple.dart';
import 'package:boquilahub/src/resources/objects.dart';

class ProcessingPage extends StatefulWidget {
  const ProcessingPage({super.key, required this.title});

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
                file.path.toLowerCase().endsWith('.jpeg'))
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

      bool isPicture = file.path.endsWith(".jpg") | file.path.endsWith(".JPG") | file.path.endsWith(".jpeg") | file.path.endsWith(".JPEG");
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
            if (isfolderselected)  Text("$nProcessed imágenes procesadas"),        
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
                    animalDataList = jsonList
                        .map((json) => BBox.fromJson(json))
                        .toList();

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
                child:
                    BoxImg(file: File(jpgFile), listBBox: animalDataList),
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

final ButtonStyle botoncitostyle = ElevatedButton.styleFrom(
  foregroundColor: currentcolors[0],
  backgroundColor: currentcolors[4],
  minimumSize: const Size(100, 45),
  padding: const EdgeInsets.symmetric(horizontal: 16),
  shape: const RoundedRectangleBorder(
    borderRadius: BorderRadius.all(Radius.circular(10)),
  ),
);



// class BoxImage extends StatelessWidget {
//   final List<BBox> listBBox;
//   final Image image;

//   const BoxImage({required this.listBBox, required this.image, Key? key})
//       : super(key: key);

//   @override
//   Widget build(BuildContext context) {
//     return Stack(
//       children: [
//         image,
//         for (var data in listBBox)
//           Positioned(
//             left: data.x1,
//             top: data.y1,
//             width: data.x2 - data.x1,
//             height: data.y2 - data.y1,
//             // bottom: data.x2,
//             // right: data.y2,
//             child: Container(
//               decoration: BoxDecoration(
//                 border: Border.all(
//                   color: Colors.red, // Change color as needed
//                   width: 2.0,
//                 ),
//               ),
//               // child: Center(
//               //   child: Text(
//               //     data.label,
//               //     style: const TextStyle(
//               //       fontSize: 24,
//               //       color: Colors.red, // Change color as needed
//               //       fontWeight: FontWeight.bold,
//               //     ),
//               //   ),
//               // ),
//             ),
//           ),
//       ],
//     );
//   }
// }

// class BoxImage2 extends StatelessWidget {
//   final List<BBox> listBBox;
//   final Image image;

//   const BoxImage2({required this.listBBox, required this.image, Key? key})
//       : super(key: key);

//   Future<Size> getsize(Image image) {
//     Completer<Size> completer = Completer();
//     image.image.resolve(ImageConfiguration()).addListener(
//       ImageStreamListener(
//         (ImageInfo image, bool synchronousCall) {
//           Size size =
//               Size(image.image.width.toDouble(), image.image.height.toDouble());
//           completer.complete(size);
//         },
//       ),
//     );
//     return completer.future;
//   }

//   @override
//   Widget build(BuildContext context) {
//     const i_w = 576;
//     const i_h = 432;

//     return FutureBuilder(
//         future: getsize(image),
//         builder: (context, snapshot) {
//           Size? _size = snapshot.data;
//           print("size: ");
//           print(_size);
//           // double ratio_w = i_w / MediaQuery.of(context).size.width;
//           // double ratio_h = i_h / MediaQuery.of(context).size.height;
//           double ratio_w = i_w / _size!.width;
//           double ratio_h = i_h / _size.height;
//           print("ratios: ");
//           print(ratio_h);
//           print(ratio_w);
//           print("media query");
//           print(MediaQuery.of(context).size.width);
//           print(MediaQuery.of(context).size.height);

//           return SizedBox(
//             width: i_w.toDouble(),
//             height: i_h.toDouble(),
//             child: Stack(
//               children: [
//                 image,
//                 for (var data in listBBox)
//                   Positioned(
//                     // actually, you need to divide by the SCALE of the image,
//                     // if you can show 100% of the image, then the scale is 1.00 and you divide by 1.00
//                     left: getPositon(data)['x'] * ratio_w,
//                     top: getPositon(data)['y'] * ratio_h,
//                     width: getPositon(data)['w'] * ratio_w,
//                     height: getPositon(data)['h'] * ratio_h,
//                     child: Container(
//                       decoration: BoxDecoration(
//                         border: Border.all(
//                           color: Colors.red, // Change color as needed
//                           width: 6.0,
//                         ),
//                       ),
//                       // child: Center(
//                       //   child: Text(
//                       //     data.label,
//                       //     style: const TextStyle(
//                       //       fontSize: 24,
//                       //       color: Colors.red, // Change color as needed
//                       //       fontWeight: FontWeight.bold,
//                       //     ),
//                       //   ),
//                       // ),
//                     ),
//                   ),
//               ],
//             ),
//           );
//         });
//   }
// }

class BoxImg extends StatefulWidget {
  final File file;
  final List<BBox> listBBox;
  const BoxImg({super.key, required this.file, required this.listBBox});

  @override
  State<BoxImg> createState() => _BoxImgState();
}

class _BoxImgState extends State<BoxImg> with WidgetsBindingObserver {
  var key = GlobalKey();
  Size? redboxSize; // displayed size

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void didChangeMetrics() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      redboxSize = getRedBoxSize(key.currentContext!);
      setState(() {});
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  Future<Size> getsize(Image image) {
    Completer<Size> completer = Completer();
    image.image.resolve(const ImageConfiguration()).addListener(
      ImageStreamListener(
        (ImageInfo image, bool synchronousCall) {
          Size size =
              Size(image.image.width.toDouble(), image.image.height.toDouble());
          completer.complete(size);
        },
      ),
    );
    return completer.future;
  }

  void delay(int i) {
    Future.delayed(Duration(milliseconds: i));
  }

  @override
  Widget build(BuildContext context) {
    Image img = Image.file(widget.file, key: key);
    if (redboxSize == null) {
      return FutureBuilder(
        future: Future.delayed(const Duration(
            milliseconds: 200)), // Introduce a delay of 2000 milliseconds
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            // While waiting, you can show a loading indicator or any other widget.
            return img;
          } else {
            // After the delay, you can return the desired widget.
            didChangeMetrics();
            return img;
          }
        },
      );
    } else {
      return Center(
        child: FutureBuilder(
            future: getsize(img),
            builder: (context, snapshot) {
              bool conA = !snapshot.hasData;
              bool conB = redboxSize == null;
              print("Conditions 1");
              print(redboxSize);
              print(conA);
              print(conB);
              if (conB || conA) {
                return const CircularProgressIndicator();
              } else {
                print("Conditions 2");
                print(redboxSize);
                print(conA);
                print(conB);
                Size? size = snapshot.data;
                double ratioW = redboxSize!.width / size!.width;
                double ratioH = redboxSize!.height / size.height;
                // print(ratioW);
                return Stack(
                  children: [
                    img,
                    for (var data in widget.listBBox)
                      Positioned(
                        left: data.x1 * ratioW,
                        top: data.y1 * ratioH,
                        width: (data.x2 - data.x1) * ratioW,
                        height: (data.y2 - data.y1) * ratioH,
                        child: Container(
                          decoration: BoxDecoration(
                            border: Border.all(
                              color: Colors.red, // Change color as needed
                              width: 2.0,
                            ),
                          ),
                        ),
                      ),
                  ],
                );
              }
            }),
      );
    }
  }

  Size getRedBoxSize(BuildContext context) {
    final box = context.findRenderObject() as RenderBox;
    return box.size;
  }
}

// class BoxImg2 extends StatefulWidget {
//   final Image img;
//   final List<BBox> listBBox;
//   const BoxImg2({super.key, required this.img, required this.listBBox});

//   @override
//   State<BoxImg2> createState() => _BoxImg2State();
// }

// class _BoxImg2State extends State<BoxImg2> with WidgetsBindingObserver {
//   var key;
//   Size? redboxSize; // displayed size

//   @override
//   void initState() {
//     super.initState();
//     WidgetsBinding.instance.addObserver(this);
//   }

//   @override
//   void didChangeMetrics() {
//     WidgetsBinding.instance.addPostFrameCallback((_) {
//       redboxSize = getRedBoxSize(key.currentContext!);
//       setState(() {});
//     });
//   }

//   @override
//   void dispose() {
//     WidgetsBinding.instance.removeObserver(this);
//     super.dispose();
//   }

//   Future<Size> getsize(Image image) {
//     Completer<Size> completer = Completer();
//     image.image.resolve(const ImageConfiguration()).addListener(
//       ImageStreamListener(
//         (ImageInfo image, bool synchronousCall) {
//           Size size =
//               Size(image.image.width.toDouble(), image.image.height.toDouble());
//           completer.complete(size);
//         },
//       ),
//     );
//     return completer.future;
//   }

//   void delay(int i) {
//     Future.delayed(Duration(milliseconds: i));
//   }

//   @override
//   Widget build(BuildContext context) {
//     key = widget.img.key;
//     getRedBoxSize(key.currentContext!);
//     print(redboxSize);
//     print(redboxSize);
//     print(redboxSize);
//     return Center(
//       child: Stack(
//         children: [
//           widget.img,
//           FutureBuilder(
//               future: getsize(widget.img),
//               builder: (context, snapshot) {
//                 getRedBoxSize(key.currentContext!);
//                 bool conA = !snapshot.hasData;
//                 bool conB = redboxSize == null;
//                 print("Conditions 1");
//                 print(redboxSize);
//                 print(conA);
//                 print(conB);
//                 if (conB || conA) {
//                   return const CircularProgressIndicator();
//                 } else {
//                   print("Conditions 2");
//                   print(redboxSize);
//                   print(conA);
//                   print(conB);
//                   Size? size = snapshot.data;
//                   double ratioW = redboxSize!.width / size!.width;
//                   double ratioH = redboxSize!.height / size.height;
//                   // print(ratioW);
//                   return Stack(
//                     children: [
//                       for (var data in widget.listBBox)
//                         Positioned(
//                           left: data.x1 * ratioW,
//                           top: data.y1 * ratioH,
//                           width: (data.x2 - data.x1) * ratioW,
//                           height: (data.y2 - data.y1) * ratioH,
//                           child: Container(
//                             decoration: BoxDecoration(
//                               border: Border.all(
//                                 color: Colors.red, // Change color as needed
//                                 width: 2.0,
//                               ),
//                             ),
//                           ),
//                         ),
//                     ],
//                   );
//                 }
//               }),
//         ],
//       ),
//     );
//   }

//   Size getRedBoxSize(BuildContext context) {
//     final box = context.findRenderObject() as RenderBox;
//     return box.size;
//   }
// }

// getPositon(BBox data) {
//   Map<String, double> position = {
//     'x': (data.x1 + data.x2) / 2,
//     'y': (data.y1 + data.y2) / 2,
//     'w': data.x2 - data.x1,
//     'h': data.y2 - data.y1,
//   };
//   print(position);
//   return position;
// }

// getNPosition(BBox data, c) {
//   double screenWidth = MediaQuery.of(c).size.width;
//   double screenHeight = MediaQuery.of(c).size.height;

//   Map<String, double> position = {
//     'x': ((data.x1 + data.x2) / 2) / screenWidth,
//     'y': ((data.y1 + data.y2) / 2) / screenHeight,
//     'w': (data.x2 - data.x1) / screenWidth,
//     'h': (data.y2 - data.y1) / screenHeight,
//   };
//   print(position);
//   return position;
// }

// class Home extends StatefulWidget {
//   @override
//   HomeState createState() => HomeState();
// }

// class HomeState extends State<Home> with WidgetsBindingObserver {
//   var key = GlobalKey();
//   Size? redboxSize;

//   @override
//   void initState() {
//     super.initState();
//     WidgetsBinding.instance.addObserver(this);
//   }

//   @override
//   void didChangeMetrics() {
//     WidgetsBinding.instance.addPostFrameCallback((_) {
//       redboxSize = getRedBoxSize(key.currentContext!);
//       setState(() {});
//     });
//   }

//   @override
//   void dispose() {
//     WidgetsBinding.instance.removeObserver(this);
//     super.dispose();
//   }

//   @override
//   Widget build(BuildContext context) {
//     return Column(
//       children: [
//         Container(
//           color: Colors.redAccent,
//           child: Image.file(
//               File(
//                   "C:/Users/Fincho/Desktop/git projects/Boquila-paper/Comparison/3.JPG"),
//               key: key),
//         ),
//         if (redboxSize != null) Text('$redboxSize'),
//       ],
//     );
//   }

//   Size getRedBoxSize(BuildContext context) {
//     final box = context.findRenderObject() as RenderBox;
//     return box.size;
//   }
// }
