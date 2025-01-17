import 'dart:io';
import 'package:boquilahub/src/resources/utils.dart';
import 'package:csv/csv.dart';
import 'package:flutter/material.dart';

class BBox {
  double x1, y1, x2, y2;
  String label;
  double confidence;

  BBox(this.x1, this.y1, this.x2, this.y2, this.label, this.confidence);

  // This will take the Input from Rust and process it to return a Bounding Box
  factory BBox.fromJson(List<dynamic> json, AI ai) {
    return BBox(
      json[0].toDouble(),
      json[1].toDouble(),
      json[2].toDouble(),
      json[3].toDouble(),
      ai.classes[json[4] as int],
      json[5].toDouble(),
    );
  }

  @override
  String toString() {
    return '$x1,$y1,$x2,$y2,$label,$confidence';
  }

  String strconf() {
    // Round the confidence to two decimal places
    return confidence.toStringAsFixed(2);
  }
}

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

class AI {
  final String name;
  final String description;
  final String colorCode; // "terra", "fire", "aqua"
  final List<String> classes;

  const AI(this.name, this.description, this.colorCode, this.classes);

  String getPath() {
    return "models/$name.bq";
  }
}

List<AI> listAIs = const <AI>[
  AI("boquilanet-gen", '🖼️ Ánimales (genérico)', "terra", boquilanetgenClasses),
  AI("boquilanet-cl", '🖼️ Ánimales (especies)', "terra", boquilanetclClasses)
];

AI getAIByDescription(String description) {
  // Function to get AI object by description
  return listAIs.firstWhere((ai) => ai.description == description);
}

class EP {
  final String name;
  final String description;
  final Widget widget; // New field

  const EP(this.name, this.description, this.widget); // Updated constructor
}

const List<EP> listEPs = <EP>[
  // EP("CPU", "Central Processing Unit", Center(child: Text("CPU"))),
  EP(
      "CUDA",
      "GPU Nvidia",
      Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Image(image: AssetImage('assets/tiny_nvidia.png')),
          Text("CUDA"),
        ],
      ))
];

const List<String> boquilanetgenClasses = <String>["animal"];

const List<String> boquilanetclClasses = <String>[
  "guanaco",
  "huemul",
  "culpeo",
  "guigna",
  "chingue",
  "pudu",
  "puma",
  "ave",
  "caballo",
  "vaca",
  "micromamiferos",
  "conejo",
  "perro"
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
