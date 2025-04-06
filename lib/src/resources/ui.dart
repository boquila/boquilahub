import 'package:flutter/material.dart';
import 'package:bitsdojo_window/bitsdojo_window.dart';

const List<Color> terra = <Color>[
  Color.fromARGB(255, 232, 252, 207),
  Color.fromARGB(255, 150, 224, 114),
  Color.fromARGB(255, 61, 163, 93),
  Color.fromARGB(255, 62, 137, 20),
  Color.fromARGB(255, 19, 70, 17)
];

final buttonColors = WindowButtonColors(
    iconNormal: const Color(0xFF805306),
    mouseOver: const Color(0xFFF6A00C),
    mouseDown: const Color(0xFF805306),
    iconMouseOver: const Color(0xFF805306),
    iconMouseDown: const Color(0xFFFFD500));

final closeButtonColors = WindowButtonColors(
    mouseOver: const Color(0xFFD32F2F),
    mouseDown: const Color(0xFFB71C1C),
    iconNormal: const Color(0xFF805306),
    iconMouseOver: Colors.white);

class WindowButtons extends StatefulWidget {
  const WindowButtons({super.key});

  @override
  State<WindowButtons> createState() => _WindowButtonsState();
}

class _WindowButtonsState extends State<WindowButtons> {
  void maximizeOrRestore() {
    setState(() {
      appWindow.maximizeOrRestore();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        MinimizeWindowButton(colors: buttonColors),
        appWindow.isMaximized
            ? RestoreWindowButton(
                colors: buttonColors,
                onPressed: maximizeOrRestore,
              )
            : MaximizeWindowButton(
                colors: buttonColors,
                onPressed: maximizeOrRestore,
              ),
        CloseWindowButton(
          colors: closeButtonColors,
        ),
      ],
    );
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
                centerTitle: true, title: title, backgroundColor: terra[2]),
            body: Center(child: child),
          ),
        ),
      ),
      child: child,
    );
  }
}