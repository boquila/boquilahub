import 'dart:async';
import 'package:boquilahub/src/resources/palettes.dart';
import 'package:flutter/material.dart';
import 'objects.dart';
import 'dart:io';

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

  // void delay(int i) {
  //   Future.delayed(Duration(milliseconds: i));
  // }

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
    return Stack(
      children: [
        delayedimg(img),
        if (redboxSize != null)
          FutureBuilder(
            future: getsize(img),
            builder: (context, snapshot) {
              bool conA = !snapshot.hasData;
              bool conB = redboxSize == null;
              // print("Conditions 1");
              // print(redboxSize);
              // print(conA);
              // print(conB);
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
                    for (BBox data in widget.predImg.listbbox)
                      Positioned(
                        left: data.x1 * ratioW,
                        top: data.y1 * ratioH,
                        width: (data.x2 - data.x1) * ratioW,
                        height: (data.y2 - data.y1) * ratioH,
                        child: Container(
                          decoration: BoxDecoration(
                            border: Border.all(
                              color: bboxColors[getIndexColor(data.label)],
                              width: 2.0,
                            ),
                          ),
                        ),
                      ),
                    for (BBox data in widget.predImg.listbbox)
                      Positioned(
                        left: data.x1 * ratioW,
                        top: (data.y1 * ratioH) - 16,
                        width: data.x2 * ratioW,
                        height: data.y2 + 10 * ratioH,
                        child: Align(
                          alignment: Alignment.topLeft,
                          child: Container(
                            color: bboxColors[getIndexColor(data.label)],
                            child: Padding(
                              padding: const EdgeInsets.all(2.0),
                              child: Text(
                                "${data.label} ${data.strconf()}",
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

// Very inefficient
int getIndexColor(String classValue) {
  if (boquilanetgenClasses.contains(classValue)) {
    return boquilanetgenClasses.indexOf(classValue);
  } else if (boquilanetclClasses.contains(classValue)) {
    return boquilanetclClasses.indexOf(classValue);
  } else if (boquilafireClasses.contains(classValue)) {
    return boquilafireClasses.indexOf(classValue);
  } else {
    // Return a default index if the class is not found in any list
    return 0;
  }
}

// class ImageScreen extends StatelessWidget {
//   final PredImg predImg;
//   final Widget child;

//   const ImageScreen({required this.child, required this.predImg, super.key});

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(
//           centerTitle: true,
//           title: Text(predImg.getFilename()),
//           backgroundColor: Colors.teal),
//       body: child,
//     );
//   }
// }

class ClickableImage extends StatelessWidget {
  final PredImg predImg;
  final Widget child;

  const ClickableImage({required this.child, required this.predImg, super.key});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () => Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => Scaffold(
            appBar: AppBar(
                centerTitle: true,
                title: Text(predImg.getFilename()),
                backgroundColor: Colors.teal),
            body: child,
          ),
        ),
      ),
      child: child,
    );
  }
}
