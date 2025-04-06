import 'package:boquilahub/src/resources/ui.dart';
import 'package:flutter/material.dart';
import 'package:boquilahub/src/rust/api/abstractions.dart';
import 'package:boquilahub/src/rust/api/eps.dart';

Widget epWidget(EP ep) {
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
      imgPath: "tiny_cpu.png",
      version: 0.0,
      local: true,
      dependencies: "none"),
  EP(
      name: "CUDA",
      imgPath: "tiny_nvidia.png",
      version: 12.4,
      local: true,
      dependencies: "cuDNN"),
  EP(
      name: "BoquilaHUB Remoto",
      imgPath: "tiny_boquila.png",
      version: 0.0,
      local: false,
      dependencies: "none"),
];

Widget aiWidget(AI value) {
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

String cudaText(double cudaversion) {
  if (cudaversion == 12.8) {
    return "se encontró la versión correcta";
  } else if (cudaversion == 0) {
    return "no se encontró una versión";
  } else {
    String text = cudaversion.toString();
    return "se encontró versión $text";
  }
}

class ClickAbleWidget extends StatelessWidget {
  final Widget title;
  final Widget child;

  const ClickAbleWidget({required this.child, required this.title, super.key});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () => Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => Scaffold(
            appBar: AppBar(
                centerTitle: true, title: title, backgroundColor: terra[4]),
            body: child,
          ),
        ),
      ),
      child: child,
    );
  }
}