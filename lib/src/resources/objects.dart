import 'dart:io';
import 'package:boquilahub/src/resources/utils.dart';
import 'package:flutter/material.dart';
import 'package:boquilahub/src/rust/api/abstractions.dart';
import 'package:boquilahub/src/rust/api/eps.dart';

class PredImg {
  final String filePath;
  List<BBox> listbbox;
  bool wasprocessed;

  PredImg(this.filePath, this.listbbox, this.wasprocessed);
}


int countProcessedImages(List<PredImg> images) {
  return images.where((img) => img.wasprocessed).length;
}

bool areBoxesEmpty(List<PredImg> images) {
  for (var image in images) {
    if (image.listbbox.isNotEmpty) {
      return false;
    }
  }
  return true;
}

String getMainLabel(List<BBox> listbbox) {
  if (listbbox.isEmpty) {
    return 'no predictions';
  } else {
    final Map<String, int> labelCounts = {};

    for (var bbox in listbbox) {
      labelCounts[bbox.label] = (labelCounts[bbox.label] ?? 0) + 1;
    }

    final mainLabel =
        labelCounts.entries.reduce((a, b) => a.value > b.value ? a : b).key;

    return mainLabel;
  }
}

Widget render(predImg) {
  return ClickableImage(predImg: predImg, child: BoxImg(predImg: predImg));
}

Widget getEPWidget(EP ep) {
  return Padding(
    padding: EdgeInsets.symmetric(horizontal: 12),
    child: Row(
      children: [
        Container(
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            boxShadow: [
              BoxShadow(
                color: const Color.fromARGB(31, 85, 194, 64),
                blurRadius: 3,
                offset: Offset(0, 1),
              ),
            ],
          ),
          child: Image.asset(
            'assets/${ep.imgPath}',
            width: 32,
            height: 32,
          ),
        ),
        SizedBox(width: 12),
        Text(
          ep.name,
          style: TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w500,
            letterSpacing: 0.3,
          ),
        ),
      ],
    ),
  );
}

const List<EP> listEPs = <EP>[
  EP(
      name: "CPU",
      description: "Just your CPU",
      imgPath: "tiny_cpu.png",
      version: 0.0,
      local: true,
      dependencies: "none"),
  EP(
      name: "CUDA",
      description: "NVIDIA GPU",
      imgPath: "tiny_nvidia.png",
      version: 12.4,
      local: true,
      dependencies: "cuDNN"),
  EP(
      name: "BoquilaHUB Remoto",
      description: "Sesión remota de BoquilaHUB",
      imgPath: "tiny_boquila.png",
      version: 0.0,
      local: false,
      dependencies: "none"),
];

Widget getAIwidget(AI value) {
  return Tooltip(
    message: value.classes.join(', '),
    child: Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Row(
          children: [
            const Text('🖼️ '),
            Text(value.name),
          ],
        ),
        if (value.classes.isNotEmpty)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
            decoration: BoxDecoration(
              color: Colors.grey.withValues(alpha: 0.2),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Text(
              'classes: ${value.classes.length}',
              style: TextStyle(
                fontSize: 12,
                color: Colors.grey[600],
              ),
            ),
          ),
      ],
    ),
  );
}

Future<void> copyToFolder(List<PredImg> predImgs, String outputPath) async {
  for (PredImg predImg in predImgs) {
    final File imageFile = File(predImg.filePath);
    if (await imageFile.exists()) {
      final String mainLabel = getMainLabel(predImg.listbbox);
      String folderPath;
      if (mainLabel == 'no predictions') {
        folderPath = '$outputPath/$mainLabel';
      } else {
        folderPath = '$outputPath/$mainLabel';
      }
      final Directory folder = Directory(folderPath);
      if (!await folder.exists()) {
        await folder.create(recursive: true);
      }
      final String imageName = imageFile.path.split('\\').last;
      final File newImageFile = File('$folderPath/$imageName');
      await newImageFile.writeAsBytes(await imageFile.readAsBytes());
    }
  }
}

simpleDialog(context, String text) {
  return showDialog(
    context: context,
    builder: (context) => AlertDialog(actions: [
      ElevatedButton(
          onPressed: () {
            Navigator.pop(context);
          },
          child: const Text("Ok"))
    ], title: Text(text)),
  );
}