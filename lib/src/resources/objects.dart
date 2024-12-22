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
  final String colorCode; // "terra", "fire", "green"
  final String outputType; // "BBox", "ProbSpace"
  final List<String> classes;
  final bool available;

  const AI(this.name, this.description, this.colorCode, this.outputType,
      this.classes, this.available);

  String getPath() {
    return "models/$name.bq";
  }
}

List<AI> listAIs = const <AI>[
  AI("boquilanet-gen", '🖼️ Ánimales (genérico)', "terra", "BBox",
      boquilanetgenClasses, true),
  AI("boquilanet-cl", '🖼️ Ánimales (especies)', "terra", "BBox",
      boquilanetclClasses, true),
  // AI('yoloXl', '🖼️ Objetos (genérico)', "terra", "BBox", yoloClasses, true),
  AI("boquila-fire", "🔥 Incendios", "fire", "ProbSpace", boquilafireClasses,
      false),
  AI('boquila-bird-gen', '🔊 Aves (genérico)', "terra", "ProbSpace", [], false),
  AI('boquila-bird-cl', '🔊 Aves (especies)', "terra", "ProbSpace", [], false),
  AI('boquila-h2o', '🔊 Hídrofonos', "aqua", "ProbSpace", [], false)
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

const List<String> boquilafireClasses = <String>["fire", "smoke", "normal"];

const List<String> yoloClasses = <String>[
  'person',
  'bicycle',
  'car',
  'motorcycle',
  'airplane',
  'bus',
  'train',
  'truck',
  'boat',
  'traffic light',
  'fire hydrant',
  'stop sign',
  'parking meter',
  'bench',
  'bird',
  'cat',
  'dog',
  'horse',
  'sheep',
  'cow',
  'elephant',
  'bear',
  'zebra',
  'giraffe',
  'backpack',
  'umbrella',
  'handbag',
  'tie',
  'suitcase',
  'frisbee',
  'skis',
  'snowboard',
  'sports ball',
  'kite',
  'baseball bat',
  'baseball glove',
  'skateboard',
  'surfboard',
  'tennis racket',
  'bottle',
  'wine glass',
  'cup',
  'fork',
  'knife',
  'spoon',
  'bowl',
  'banana',
  'apple',
  'sandwich',
  'orange',
  'broccoli',
  'carrot',
  'hot dog',
  'pizza',
  'donut',
  'cake',
  'chair',
  'couch',
  'potted plant',
  'bed',
  'dining table',
  'toilet',
  'tv',
  'laptop',
  'mouse',
  'remote',
  'keyboard',
  'cell phone',
  'microwave',
  'oven',
  'toaster',
  'sink',
  'refrigerator',
  'book',
  'clock',
  'vase',
  'scissors',
  'teddy bear',
  'hair drier',
  'toothbrush'
];

// AI(
//   "boquilanet-eu",
//   "European fauna classification",
//   "terra",
//   "ProbSpace"
// ),
// AI(
//   "megadetector v5a",
//   "MegaDetector (animals, vehicles, people)",
//   "green",
//   "BBox"
// ),

Future<void> writeCsv(List<PredImg> predImgs, String outputPath) async {
  List<List<dynamic>> rows = [];

  rows.add(['File Path', 'X1', 'Y1', 'X2', 'Y2', 'Label', 'Confidence']);

  for (var predImg in predImgs) {
    for (var bbox in predImg.listbbox) {
      rows.add([
        predImg.filePath,
        bbox.x1,
        bbox.y1,
        bbox.x2,
        bbox.y2,
        bbox.label,
        bbox.confidence
      ]);
    }
  }

  String csv = const ListToCsvConverter().convert(rows);

  final file = File(outputPath);
  await file.writeAsString(csv);
}

Future<void> writeCsv2(List<PredImg> predImgs, String outputPath) async {
  final List<List<dynamic>> rows = [];

  rows.add(['File Path', 'n', 'observaciones']);

  for (var predImg in predImgs) {
    final List<List<dynamic>> bboxRows = [];
    final Set<String> labels = {};

    for (BBox bbox in predImg.listbbox) {
      bboxRows.add([
        bbox.x1,
        bbox.y1,
        bbox.x2,
        bbox.y2,
        bbox.label,
        bbox.confidence,
      ]);
      labels.add(bbox.label);
    }

    rows.add([
      predImg.filePath,
      bboxRows.length,
      labels.join(', '), // Joining all labels with a comma
    ]);
  }

  final String csv = const ListToCsvConverter().convert(rows);

  final File file = File(outputPath);
  await file.writeAsString(csv);
}

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
