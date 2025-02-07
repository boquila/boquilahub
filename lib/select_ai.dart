import 'package:boquilahub/src/resources/hardware_dep.dart';
import 'package:flutter/material.dart';
import 'src/resources/objects.dart';
import 'package:boquilahub/src/rust/api/abstractions.dart';
import 'package:boquilahub/src/rust/api/eps.dart';
import 'package:boquilahub/src/rust/api/rest.dart';
import 'package:flutter/gestures.dart';
import 'package:flutter/services.dart';

class SelectAIPage extends StatefulWidget {
  final Function(AI?) aicallback;
  final Function(EP) epcallback;
  final Function(String?) urlcallback;
  final List<Color> currentcolors;
  final EP currentEP;
  final AI? currentAI;
  final List<AI> listAIs;
  const SelectAIPage(
      {super.key,
      required this.aicallback,
      required this.epcallback,
      required this.urlcallback,
      required this.currentcolors,
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

  @override
  Widget build(BuildContext context) {
    TextStyle textito =
        TextStyle(color: widget.currentcolors[4], fontWeight: FontWeight.bold);

    ButtonStyle botoncitostyle = ElevatedButton.styleFrom(
      foregroundColor: widget.currentcolors[0],
      backgroundColor: widget.currentcolors[4],
      minimumSize: const Size(100, 45),
      padding: const EdgeInsets.symmetric(horizontal: 16),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.all(Radius.circular(10)),
      ),
    );

    return Column(
      children: [
        Text("Selecciona una IA", style: textito),
        const SizedBox(height: 10),
        DropdownButton<String>(
          value: aiDropDownValue,
          icon: const Icon(Icons.search),
          elevation: 4,
          style: TextStyle(color: widget.currentcolors[4]),
          underline: Container(
            height: 0.25,
            width: 5,
            color: widget.currentcolors[2],
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
              child: getAIwidget(value),
            );
          }).toList(),
        ),
        const SizedBox(height: 30),
        Text("Procesador", style: textito),
        const SizedBox(height: 10),
        DropdownButton<String>(
          value: epDropdownValue,
          icon: const Icon(Icons.search),
          elevation: 1,
          style: TextStyle(color: widget.currentcolors[4]),
          underline: Container(
            height: 0.25,
            width: 5,
            color: widget.currentcolors[2],
          ),
          onChanged: (String? value) async {
            if (value == "CUDA") {
              EP tempEP = getEpByName(listEps: listEPs, name: value!);
              double cudaVersion =
                  await ExecutionProviders.cuda(tempEP).getVersion();
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
              // TODO: It needs to check that URL is actually a good BoquilaHUB API
              if (isAPIdeployed) {
                if (context.mounted) {
                  simpleDialog(context,
                      "No puedes elegir está opción, \nya que estás desplegando una API");
                }
              } else if (url != null) {
                bool isapigood = await checkBoquilaHubApi(url: url);
                if (isapigood){
                  setState(() {
                  EP tempep = getEpByName(listEps: listEPs, name: value!);
                  widget.epcallback(tempep);
                  widget.urlcallback(url);
                  epDropdownValue = value;
                });
                } else {
                  if (context.mounted){
                  simpleDialog(context, "Hubo un error, por favor verifica la URL");
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
              child: getEPWidget(value),
            );
          }).toList(),
        ),
        const SizedBox(height: 30),
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
                  text: 'http://${getIp()}:8791',
                  style: TextStyle(
                    color: Colors.blue,
                  ),
                  recognizer: TapGestureRecognizer()
                    ..onTap = () {
                      Clipboard.setData(
                          ClipboardData(text: 'http://${getIp()}:8791'));
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
        if (apierror) Text("Ocurrió un error")
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
                  Uri.tryParse(enteredUrl)?.hasAbsolutePath == true) {
                Navigator.of(context).pop(enteredUrl); // Return the URL
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
