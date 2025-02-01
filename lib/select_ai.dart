import 'package:boquilahub/src/resources/hardware_dep.dart';
import 'package:flutter/material.dart';
import 'src/resources/objects.dart';

import 'package:boquilahub/src/rust/api/abstractions.dart';
import 'package:boquilahub/src/rust/api/eps.dart';

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

  @override
  Widget build(BuildContext context) {
    TextStyle textito =
        TextStyle(color: widget.currentcolors[4], fontWeight: FontWeight.bold);

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
            if (currentAI != null) {
              if (value == "CUDA") {
                EP tempEP = getEpByName(listEps: listEPs, name: value!);
                double cudaVersion =
                    await ExecutionProviders.cuda(tempEP).getVersion();
                bool iscudnnAvailable =
                    true; // TODO: implement isCUDNNAvailable
                if (cudaVersion == requiredCuda && iscudnnAvailable) {
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
              }
              setState(() {
                widget.aicallback(currentAI!, currentEP);
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
      ],
    );
  }
}
