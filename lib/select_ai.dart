import 'package:flutter/material.dart';
import 'src/resources/ui.dart';
import 'package:boquilahub/src/rust/api/abstractions.dart';
import 'package:boquilahub/src/rust/api/eps.dart';
import 'package:boquilahub/src/rust/api/rest.dart';
import 'package:flutter/gestures.dart';
import 'package:flutter/services.dart';

class SelectAIPage extends StatefulWidget {
  final Function(AI?) aicallback;
  final Function(EP) epcallback;
  final Function(String?) urlcallback;
  final EP currentEP;
  final AI? currentAI;
  final List<AI> listAIs;
  const SelectAIPage(
      {super.key,
      required this.aicallback,
      required this.epcallback,
      required this.urlcallback,
      required this.currentEP,
      required this.currentAI,
      required this.listAIs});

  @override
  State<SelectAIPage> createState() => _SelectAIPageState();
}

class _SelectAIPageState extends State<SelectAIPage> {
  String epDropdownValue = listEPs.first.name;
  String? aiDropDownValue;
  bool isAPIdeployed = false;
  bool apierror = false;
  final String ip = getIp(); // Call getIp() only once

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

  @override
  Widget build(BuildContext context) {
    TextStyle textito = TextStyle(color: terra[4], fontWeight: FontWeight.bold);

    ButtonStyle botoncitostyle = ElevatedButton.styleFrom(
      foregroundColor: terra[0],
      backgroundColor: terra[4],
      minimumSize: const Size(100, 45),
      padding: const EdgeInsets.symmetric(horizontal: 16),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.all(Radius.circular(10)),
      ),
    );

    return Column(
      children: [
        // SECTION: AI 
        Text("Selecciona una IA", style: textito),
        const SizedBox(height: 10),
        DropdownButton<String>(
          value: aiDropDownValue,
          icon: const Icon(Icons.search),
          elevation: 4,
          style: TextStyle(color: terra[4]),
          underline: Container(
            height: 0.25,
            width: 5,
            color: terra[2],
          ),
          onChanged: (String? value) {
            if (true) {
              setState(() {
                AI tempAI = getAiByDescription(
                    listAis: widget.listAIs, description: value!);
                widget.aicallback(tempAI);
                aiDropDownValue = value;
              });
            }
          },
          items: widget.listAIs.map<DropdownMenuItem<String>>((AI value) {
            return DropdownMenuItem<String>(
              value: value.name,
              child: aiWidget(value),
            );
          }).toList(),
        ),
        const SizedBox(height: 30),
        // SECTION: EP selection
        // at the end, it will call setmodel() with the new EP data
        Text("Procesador", style: textito),
        const SizedBox(height: 10),
        DropdownButton<String>(
          value: epDropdownValue,
          icon: const Icon(Icons.search),
          elevation: 1,
          style: TextStyle(color: terra[4]),
          underline: Container(
            height: 0.25,
            width: 5,
            color: terra[2],
          ),
          onChanged: (String? value) async {
            if (value == "CUDA") {
              EP tempEP = getEpByName(listEps: listEPs, name: value!);
              double cudaVersion = await getEpVersion(provider: tempEP);
              bool iscudnnAvailable = true; // TODO: implement isCUDNNAvailable
              if (cudaVersion == 12.8 && iscudnnAvailable) {
                setState(() {
                  widget.epcallback(tempEP);
                  epDropdownValue = value;
                });
              } else {
                if (!context.mounted) return;
                String cudatext = cudaText(cudaVersion);
                showDialog(
                  context: context,
                  builder: (context) => AlertDialog(
                    actions: [
                      ElevatedButton(
                          onPressed: () {
                            Navigator.pop(context);
                          },
                          child: const Text("Ok"))
                    ],
                    content: Text(
                        "Se requiere \n- CUDA 12.8, $cudatext\n- cuDNN 9.7,\n- Tarjeta gráfica Nvidia 7xx o superior \n\n Por favor verificar que todos los requisitos se cumplan"),
                  ),
                );
              }
            } else if (value == "CPU") {
              setState(() {
                EP tempep = getEpByName(listEps: listEPs, name: value!);
                widget.epcallback(tempep);
                epDropdownValue = value;
              });
            } else if (value == "BoquilaHUB Remoto") {
              String? url = await showUrlInputDialog(context);
              if (isAPIdeployed) {
                if (context.mounted) {
                  simpleDialog(context,
                      "No puedes elegir está opción, \nya que estás desplegando una API");
                }
              } else if (url != null) {
                bool isapigood = await checkBoquilaHubApi(url: url);
                if (isapigood) {
                  setState(() {
                    EP tempep = getEpByName(listEps: listEPs, name: value!);
                    widget.epcallback(tempep);
                    widget.urlcallback(url);
                    epDropdownValue = value;
                  });
                } else {
                  if (context.mounted) {
                    simpleDialog(
                        context, "Hubo un error, por favor verifica la URL");
                  }
                }
              }
            }
            if (widget.currentAI != null) {
              setState(() {
                widget.aicallback(widget.currentAI!);
              });
            }
          },
          items: listEPs.map<DropdownMenuItem<String>>((EP value) {
            return DropdownMenuItem(
              value: value.name,
              child: epWidget(value),
            );
          }).toList(),
        ),
        const SizedBox(height: 30),
        // SECTION: API deployment
        Text("API", style: textito),
        const SizedBox(height: 10),
        if (!isAPIdeployed && widget.currentEP.local)
          Tooltip(
            textAlign: TextAlign.center,
            waitDuration: const Duration(milliseconds: 500),
            message:
                "Esto permite que otros dispositivos \nen su red local puedan procesar datos\n a través de esta aplicación",
            child: ElevatedButton(
              style: botoncitostyle,
              onPressed: () async {
                if (widget.currentAI == null) {
                  simpleDialog(context, "Primero, elige una IA");
                } else {
                  try {
                    setState(() {
                      isAPIdeployed = true;
                      apierror = false;
                    });
                    await runApi();
                  } catch (e) {
                    setState(() {
                      isAPIdeployed = false;
                      apierror = true;
                    });
                  }
                }
              },
              child: const Text('Desplegar'),
            ),
          ),
        if (isAPIdeployed || !widget.currentEP.local)
          Tooltip(
            textAlign: TextAlign.center,
            waitDuration: const Duration(milliseconds: 500),
            message:
                "Esto permite que otros dispositivos \nen su red local puedan procesar datos\n a través de esta aplicación",
            child: ElevatedButton(
              onPressed: null,
              child: const Text('Desplegar'),
            ),
          ),
        if (isAPIdeployed)
          Text.rich(
            TextSpan(
              text: 'API desplegada en \nURL local: ',
              children: [
                TextSpan(
                  text: 'http://$ip:8791',
                  style: TextStyle(
                    color: Colors.blue,
                  ),
                  recognizer: TapGestureRecognizer()
                    ..onTap = () {
                      Clipboard.setData(ClipboardData(text: 'http://$ip:8791'));
                      // Optional: Show a snackbar or toast to indicate copying
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(content: Text('URL copiada al portapapeles')),
                      );
                    },
                ),
              ],
            ),
            textAlign: TextAlign.center,
          ),
        if (apierror) Text("Ocurrió un error"),
        const SizedBox(height: 30),
        // SECTION: Extra configs
      ],
    );
  }
}

Future<String?> showUrlInputDialog(BuildContext context) async {
  final TextEditingController urlController = TextEditingController();

  return showDialog<String?>(
    context: context,
    barrierDismissible: false, // User must type the URL and submit
    builder: (BuildContext context) {
      return AlertDialog(
        title: Text('Ingresa la URL'),
        content: TextField(
          controller: urlController,
          keyboardType: TextInputType.url,
          decoration: InputDecoration(
            hintText: 'http://127.0.0.1:8971',
          ),
        ),
        actions: <Widget>[
          TextButton(
            child: Text('Cancelar'),
            onPressed: () {
              Navigator.of(context).pop(null); // Return null on cancel
            },
          ),
          ElevatedButton(
            child: Text('Enviar'),
            onPressed: () {
              final String enteredUrl = urlController.text.trim();
              if (enteredUrl.isNotEmpty &&
                  (Uri.tryParse(enteredUrl)?.hasScheme == true &&
                      (enteredUrl.startsWith('http://') ||
                          enteredUrl.startsWith('https://')))) {
                // Valid URL
                Navigator.of(context).pop(removeTrailingSlash(enteredUrl));
              } else {
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('Por favor, ingresa una URL válida')),
                );
              }
            },
          ),
        ],
      );
    },
  );
}

String removeTrailingSlash(String input) {
  if (input.isNotEmpty && input.endsWith('/')) {
    return input.substring(0, input.length - 1);
  }
  return input;
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
