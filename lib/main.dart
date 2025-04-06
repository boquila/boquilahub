import 'package:boquilahub/src/resources/objects.dart';
import 'package:flutter/material.dart';
import 'package:bitsdojo_window/bitsdojo_window.dart';
import 'package:boquilahub/src/rust/frb_generated.dart';
import 'package:boquilahub/src/rust/api/abstractions.dart';
import 'processing.dart';
import 'select_ai.dart';
import 'src/resources/ui.dart';
import 'package:boquilahub/src/rust/api/inference.dart';
import 'package:boquilahub/src/rust/api/bq.dart';
import 'package:boquilahub/src/rust/api/eps.dart';

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

class CoreApp extends StatefulWidget {
  final List<AI> listAIs;
  const CoreApp({super.key, required this.listAIs});

  @override
  State<CoreApp> createState() => _CoreAppState();
}

class _CoreAppState extends State<CoreApp> {
  bool isLoadingAI = false;
  AI? currentAI;
  EP currentEP = listEPs[0]; // CPU as default
  String? remoteUrl;

  @override
  Widget build(BuildContext context) {
    Color sidebarColor = terra[1];
    Color backgroundStartColor = terra[0];
    Color backgroundEndColor = terra[1];

    changeAI(AI? newAI) async {
      setState(() {
        isLoadingAI = true;
        currentAI = newAI;
      });

      await setModel(value: await currentAI!.getPath(), ep: currentEP);

      setState(() {
        isLoadingAI = false;
      });
    }

    changeEP(EP newep) {
      setState(() {
        currentEP = newep;
      });
    }

    changeURL(String? url) {
      setState(() {
        remoteUrl = url;
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
          color: Color(0xFF805306),
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
                        epcallback: changeEP,
                        urlcallback: changeURL,
                        listAIs: widget.listAIs,
                        currentEP: currentEP,
                        currentAI: currentAI,
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
                    ProcessingPage(
                      currentai: currentAI,
                      currentep: currentEP,
                      url: remoteUrl,
                    )
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

