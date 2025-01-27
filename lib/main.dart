import 'package:flutter/material.dart';
import 'package:bitsdojo_window/bitsdojo_window.dart';
import 'package:boquilahub/src/rust/frb_generated.dart';
import 'package:boquilahub/src/rust/api/abstractions.dart';
// import 'package:flutter/services.dart';
import 'processing.dart';
import 'src/resources/palettes.dart';
import 'select_ai.dart';
import 'src/resources/windows.dart';
import 'package:boquilahub/src/rust/api/inference.dart';
import 'package:boquilahub/src/rust/api/bq.dart';
import 'package:boquilahub/src/rust/api/eps.dart';
// import 'src/resources/hardware_dep.dart';

Future<void> main() async {
  await RustLib.init();
  final List<AI> listAIs = await getBqs();
  runApp(CoreApp(listAIs: listAIs));

  doWhenWindowReady(() {
    final win = appWindow;
    const initialSize = Size(800, 720);
    win.minSize = initialSize;
    win.size = initialSize;
    win.alignment = Alignment.center;
    win.title = "BoquilaHUB";
    win.show();
  });
}

const borderColor = Color(0xFF805306);

class CoreApp extends StatefulWidget {
  final List<AI> listAIs;
  const CoreApp({super.key, required this.listAIs});

  @override
  State<CoreApp> createState() => _CoreAppState();
}

class _CoreAppState extends State<CoreApp> {
  List<Color> currentcolors = terra;
  bool isLoadingAI = false;
  late AI currentAI = widget.listAIs.first;
  @override
  Widget build(BuildContext context) {
    // AI currentAI = widget.listAIs.first;
    TextStyle textito =
        TextStyle(color: currentcolors[4], fontWeight: FontWeight.bold);
    Color sidebarColor = currentcolors[1];
    Color backgroundStartColor = currentcolors[0];
    Color backgroundEndColor = currentcolors[1];

    changeAI(AI newAI, EP ep) async {
      setState(() {
        isLoadingAI = true;
        currentAI = newAI;
      });
      await setModel(value: await currentAI.getPath(), ep: ep);
      setState(() {
        isLoadingAI = false;
      });
    }

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.green.shade200),
        useMaterial3: true,
      ),
      home: Scaffold(
        backgroundColor: Colors.black,
        body: WindowBorder(
          color: borderColor,
          width: 0,
          child: Row(
            children: [
              // lEFT SIDE
              SizedBox(
                width: 300,
                child: Container(
                  color: sidebarColor,
                  child: Column(
                    children: [
                      WindowTitleBarBox(child: MoveWindow()),
                      SelectAIPage(
                        aicallback: changeAI,
                        currentcolors: currentcolors,
                        listAIs: widget.listAIs,
                      ),
                      const SizedBox(height: 100)
                    ],
                  ),
                ),
              ),
              // RIGHT SIDE
              Expanded(
                child: Container(
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                        colors: [backgroundStartColor, backgroundEndColor],
                        stops: const [0.0, 1.0]),
                  ),
                  child: Column(children: [
                    WindowTitleBarBox(
                      child: Row(
                        children: [
                          Expanded(child: MoveWindow()),
                          const WindowButtons()
                        ],
                      ),
                    ),
                    Text("Selecciona una", style: textito),
                    ProcessingPage(
                        currentcolors: currentcolors)
                  ]),
                ),
              )
            ],
          ),
        ),
      ),
    );
  }
}
