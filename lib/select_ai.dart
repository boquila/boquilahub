import 'package:boquilahub/src/resources/hardware_dep.dart';
import 'package:flutter/material.dart';
import 'src/resources/objects.dart';

import 'package:boquilahub/src/rust/api/abstractions.dart';
import 'package:boquilahub/src/rust/api/eps.dart';

AI getAIByDescription(List<AI> listAIs, String description) {
  return listAIs.firstWhere((ai) => ai.description == description);
}

EP getEPByName(List<EP> listEPs, String name) {
  return listEPs.firstWhere((ep) => ep.name == name);
}

class SelectAIPage extends StatefulWidget {
  final Function(AI,EP) aicallback;
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
  String epvalue = listEPs.first.name;
  String? dropdownValue;
  late AI currentAI;
  late EP currentEP = listEPs[0];


  @override
  Widget build(BuildContext context) {
    TextStyle textito =
        TextStyle(color: widget.currentcolors[4], fontWeight: FontWeight.bold);
    
    return Column(
      children: [
        Text("Selecciona una IA", style: textito),
        const SizedBox(height: 10),
        DropdownButton<String>(
          value: dropdownValue,
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
                AI tempAI = getAIByDescription(widget.listAIs, value!);
                currentAI = tempAI;
                dropdownValue = value;
                widget.aicallback(currentAI, currentEP);
              });
            }
          },
          items: widget.listAIs.map<DropdownMenuItem<String>>((AI value) {
            return DropdownMenuItem<String>(
              value: value.description,
              child: Text(value.description),
            );
          }).toList(),
        ),
        const SizedBox(height: 30),
        Text("Procesador", style: textito),
        const SizedBox(height: 10),
        DropdownButton<String>(
          value: epvalue,
          icon: const Icon(Icons.search),
          elevation: 1,
          style: TextStyle(color: widget.currentcolors[4]),
          underline: Container(
            height: 0.25,
            width: 5,
            color: widget.currentcolors[2],
          ),
          onChanged: (String? value) async {
            // print(value);
            if (value == "CUDA") {              
              EP tempEP = getEPByName(listEPs, value!);
              double cudaVersion = await ExecutionProviders.cuda(tempEP).getVersion();
              print(cudaVersion);
              bool iscudnnAvailable = true; // TODO: implement isCUDNNAvailable
              if (cudaVersion == 12.8 && iscudnnAvailable) {
                print("Gotta change runtime");
                setState(() {
                  currentEP = tempEP;
                  epvalue = value;
                  widget.aicallback(currentAI, currentEP);
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
                        "Se requiere \n- CUDA 12.4, $cudatext\n- cuDNN 8.9.2.26, $cuDNNtext\n- Tarjeta gráfica Nvidia 7xx o superior \n\n Por favor verificar que todos los requisitos se cumplan"),
                  ),
                );
              }
            } else if (value == "CPU") {
              // print("Gotta change runtime");
              EP tempEP = getEPByName(listEPs, value!);
              setState(() {
                currentEP = tempEP;
                epvalue = value;
                widget.aicallback(currentAI, currentEP);
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
