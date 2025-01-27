import 'dart:io';
import 'package:boquilahub/src/resources/utils.dart';
import 'package:flutter/material.dart';
import 'package:boquilahub/src/rust/api/abstractions.dart';
import 'package:boquilahub/src/rust/api/eps.dart';

class PredImg {
  final String filePath;
  List<BBox> listbbox;

  PredImg(this.filePath, this.listbbox);

  Widget render() {
    return ClickableImage(predImg: this, child: BoxImg(predImg: this));
  }

  String getMainLabel() {
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

  String getFilename() {
    return filePath.split('\\').last;
  }
}
getEPWidget(EP ep) {
  return Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Image(image: AssetImage('assets/${ep.imgPath}')),
          Text(ep.name),
        ],
      );
}

const List<EP> listEPs = <EP>[
  EP(name: "CPU", description: "Just your CPU", imgPath: "tiny_cpu.png", version: 0.0, dependencies: "none"),
  EP(name: "CUDA", description: "NVIDIA GPU", imgPath: "tiny_nvidia.png", version: 12.4, dependencies: "cuDNN"),
  
];

Future<void> copyToFolder(List<PredImg> predImgs, String outputPath) async {
  for (PredImg predImg in predImgs) {
    final File imageFile = File(predImg.filePath);
    if (await imageFile.exists()) {
      final String mainLabel = predImg.getMainLabel();
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
