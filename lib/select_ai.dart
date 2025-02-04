import 'package:boquilahub/src/resources/hardware_dep.dart';
import 'package:flutter/material.dart';
import 'src/resources/objects.dart';
import 'package:boquilahub/src/rust/api/abstractions.dart';
import 'package:boquilahub/src/rust/api/eps.dart';
import 'package:boquilahub/src/rust/api/rest.dart';

class SelectAIPage extends StatefulWidget {
  final Function(AI, EP) aicallback;
  final List<Color> currentcolors;
  final List<AI> listAIs;
  const SelectAIPage(
      {super.key,
      required this.aicallback,
      required this.currentcolors,
      required this.listAIs});

  @override
  State<SelectAIPage> createState() => _SelectAIPageState();
}

class _SelectAIPageState extends State<SelectAIPage> {
  String epDropdownValue = listEPs.first.name;
  String? aiDropDownValue;
  AI? currentAI;
  late EP currentEP = listEPs[0]; // CPU as default
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
                currentAI = getAiByDescription(
                    listAis: widget.listAIs, description: value!);
                aiDropDownValue = value;
                widget.aicallback(currentAI!, currentEP);
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
            if (currentAI != null && value == "CUDA") {
              EP tempEP = getEpByName(listEps: listEPs, name: value!);
              double cudaVersion =
                  await ExecutionProviders.cuda(tempEP).getVersion();
              bool iscudnnAvailable = true; // TODO: implement isCUDNNAvailable
              if (cudaVersion == 12.8 && iscudnnAvailable) {
                setState(() {
                  currentEP = tempEP;
                  epDropdownValue = value;
                });
              } else {
                if (!context.mounted) return;
                String cudatext = cudaText(cudaVersion);
                String cuDNNtext = cuDNNText(iscudnnAvailable);
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
                        "Se requiere \n- CUDA 12.8, $cudatext\n- cuDNN 9.7, $cuDNNtext\n- Tarjeta gráfica Nvidia 7xx o superior \n\n Por favor verificar que todos los requisitos se cumplan"),
                  ),
                );
              }
            } else if (value == "CPU") {
              setState(() {
                currentEP = getEpByName(listEps: listEPs, name: value!);
                epDropdownValue = value;
              });
            } else if (value == "BoquilaHUB Remoto") {
              setState(() {
                currentEP = getEpByName(listEps: listEPs, name: value!);
                epDropdownValue = value;
              });
            }
            setState(() {
              widget.aicallback(currentAI!, currentEP);
            });
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
        if (!isAPIdeployed)
          Tooltip(
            textAlign: TextAlign.center,
            waitDuration: const Duration(milliseconds: 500),
            message:
                "Esto permite que otros dispositivos \nen su red local puedan procesar datos\n a través de esta aplicación",
            child: ElevatedButton(
              style: botoncitostyle,
              onPressed: () async {
                if (currentAI == null) {
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
        if (isAPIdeployed)
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
          Text(
            "API desplegada en \nURL local: http://{IP}:8791",
            textAlign: TextAlign.center,
          ),
        if (apierror) Text("Ocurrió un error")
      ],
    );
  }
}
