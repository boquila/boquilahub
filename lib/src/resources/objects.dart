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

Future<List<BBox>> readPredictionsFromFile(String inputPath) async {
  // Create expected filename based on input filepath
  
  final predictionPath = '${inputPath.substring(0, inputPath.lastIndexOf('.'))}_predictions.txt';
  final file = File(predictionPath);
  
  try {
    // Check if file exists
    if (!await file.exists()) {
      // print('No prediction file found at: $predictionPath');
      return [];
    }
    
    // Read and parse file
    final lines = await file.readAsLines();
    final List<BBox> bboxes = [];
    
    for (final line in lines) {
      final parts = line.split(' ');
      if (parts.length != 7) {
        // print('Warning: Skipping invalid line format: $line');
        continue;
      }
      
      try {
        bboxes.add(BBox(
          x1: double.parse(parts[1]), // x1
          y1: double.parse(parts[2]), // y1
          x2: double.parse(parts[3]), // x2
          y2: double.parse(parts[4]), // y2
          confidence: double.parse(parts[6]), // confidence
          classId: int.parse(parts[5]),    // classId
          label: parts[0]                // label
        ));
      } catch (e) {
        // print('Warning: Error parsing line: $line\nError: $e');
        continue;
      }
    }
    
    // print('Successfully read ${bboxes.length} predictions from: $predictionPath');
    return bboxes;
    
  } catch (e) {
    // print('Error reading predictions file: $e');
    rethrow;
  }
}

Future<void> writePredImgToFile(PredImg predImg) async {
  // Create output filename based on input filepath
  final inputPath = predImg.filePath;
  final outputPath = '${inputPath.substring(0, inputPath.lastIndexOf('.'))}_predictions.txt';
  
  // Open file for writing
  final file = File(outputPath);
  
  try {
    // Create StringBuffer for efficient string concatenation
    final buffer = StringBuffer();
    
    // Write each BBox as a line in the file
    for (final bbox in predImg.listbbox) {
      buffer.writeln('${bbox.label} ${bbox.x1} ${bbox.y1} ${bbox.x2} ${bbox.y2} ${bbox.classId} ${bbox.confidence}');
    }
    
    // Write the contents to file
    await file.writeAsString(buffer.toString());
    
    // print('Successfully wrote predictions to: $outputPath');
  } catch (e) {
    // print('Error writing to file: $e');
    rethrow;
  }
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