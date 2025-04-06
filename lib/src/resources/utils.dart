import 'dart:async';
import 'package:flutter/material.dart';
import 'dart:io';
import 'package:boquilahub/src/rust/api/abstractions.dart';

const List<Color> bboxColor = <Color>[
  Colors.red,
  Colors.deepPurple,
  Colors.lightBlueAccent,
  Colors.lightGreen,
  Colors.lime,
  Colors.orange,
  Colors.amber,
  Colors.purpleAccent,
  Colors.blue,
  Colors.deepOrange,
  Colors.purple,
  Colors.yellow,
  Colors.cyan,
  Colors.brown,
  Colors.pinkAccent,
  Colors.indigoAccent,
  Colors.teal,
  Colors.pink,
  Colors.indigo,
  Color.fromARGB(255, 128, 169, 179), // Blue Gray
  Color.fromARGB(255, 153, 102, 153), // Dark Lilac
  Color.fromARGB(255, 85, 107, 47), // Dark Olive Green
  Color.fromARGB(255, 240, 230, 140), // Khaki
  Color.fromARGB(255, 210, 180, 140), // Tan
  Color.fromARGB(255, 219, 112, 147), // Dusty Rose
  Color.fromARGB(255, 255, 218, 185), // Peach
  Color.fromARGB(255, 139, 117, 85), // Rosy Brown
  Color.fromARGB(255, 255, 160, 122), // Light Salmon
  Color.fromARGB(255, 60, 179, 113), // Medium Sea Green
  Color.fromARGB(255, 128, 0, 128), // Purple
];

const List<Color> bboxColors = [
  ...bboxColor,
  ...bboxColor,
  ...bboxColor,
];

class BoxImg extends StatefulWidget {
  final PredImg predImg;
  const BoxImg({super.key, required this.predImg});

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

  Widget delayedimg(Image img) {
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
  }

  @override
  Widget build(BuildContext context) {
    Image img = Image.file(File(widget.predImg.filePath), key: key);
    if (widget.predImg.listBbox.isEmpty) {
      return img;
    }
    return Stack(
      children: [
        delayedimg(img),
        if (redboxSize != null)
          FutureBuilder(
            future: getsize(img),
            builder: (context, snapshot) {
              bool conA = !snapshot.hasData;
              bool conB = redboxSize == null;
              if (conB || conA) {
                return const CircularProgressIndicator();
              } else {
                Size? size = snapshot.data;
                double ratioW = redboxSize!.width / size!.width;
                double ratioH = redboxSize!.height / size.height;
                return Stack(
                  clipBehavior: Clip.none,
                  children: [
                    Container(),
                    for (XYXYc data in widget.predImg.listBbox)
                      Positioned(
                        left: data.xyxy.x1 * ratioW,
                        top: data.xyxy.y1 * ratioH,
                        width: (data.xyxy.x2 - data.xyxy.x1) * ratioW,
                        height: (data.xyxy.y2 - data.xyxy.y1) * ratioH,
                        child: Container(
                          decoration: BoxDecoration(
                            border: Border.all(
                              color: bboxColors[data.xyxy.classId],
                              width: 2.0,
                            ),
                          ),
                        ),
                      ),
                    for (XYXYc data in widget.predImg.listBbox)
                      Positioned(
                        left: data.xyxy.x1 * ratioW,
                        top: (data.xyxy.y1 * ratioH) - 16,
                        width: data.xyxy.x2 * ratioW,
                        height: data.xyxy.y2 + 10 * ratioH,
                        child: Align(
                          alignment: Alignment.topLeft,
                          child: Container(
                            color: bboxColors[data.xyxy.classId],
                            child: Padding(
                              padding: const EdgeInsets.all(2.0),
                              child: Text(
                                data.strlabel(),
                                style: const TextStyle(
                                  color:
                                      Colors.white, // Set text color to white
                                ),
                              ),
                            ),
                          ),
                        ),
                      ),
                  ],
                );
              }
            },
          )
      ],
    );
  }

  Size getRedBoxSize(BuildContext context) {
    final box = context.findRenderObject() as RenderBox;
    return box.size;
  }
}