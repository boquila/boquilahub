import 'package:boquilahub/src/resources/hardware_dep.dart';
import 'package:flutter/material.dart';
import 'src/resources/objects.dart';
import 'package:boquilahub/src/rust/api/utils.dart';

class SelectAIPage extends StatefulWidget {
  final Function(AI) aicallback;
  final List<Color> currentcolors;
  const SelectAIPage(
      {super.key, required this.aicallback, required this.currentcolors});

  @override
  State<SelectAIPage> createState() => _SelectAIPageState();
}

class _SelectAIPageState extends State<SelectAIPage> {
  String dropdownValue = listAIs.first.description;
  String dropdownValue2 = listEPs.first.name;

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
            if (dropdownValue != value) {
              setState(() {
                AI tempAI = getAIByDescription(value!);
                if (tempAI.available == true) {
                  dropdownValue = value;
                  widget.aicallback(tempAI);
                } else {
                  showDialog(
                      context: context,
                      builder: (context) => AlertDialog(actions: [
                            ElevatedButton(
                                onPressed: () {
                                  Navigator.pop(context);
                                },
                                child: const Text("Ok"))
                          ], title: const Text("Sección no disponible")));
                }
              });
            }
          },
          items: listAIs.map<DropdownMenuItem<String>>((AI value) {
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
          value: dropdownValue2,
          icon: const Icon(Icons.search),
          elevation: 1,
          style: TextStyle(color: widget.currentcolors[4]),
          underline: Container(
            height: 0.25,
            width: 5,
            color: widget.currentcolors[2],
          ),
          onChanged: (String? value) async {
            print(value);
            if (value == "CUDA") {
              double cudaVersion = await getCudaVersion();
              bool iscudnnAvailable = await doescuDNNexists();
              if (cudaVersion == 12.4 && iscudnnAvailable) {
                print("Gotta change runtime");
                setState(() {
                  dropdownValue2 = value!;
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
              print("Gotta change runtime");
              setState(() {
                dropdownValue2 = value!;
              });
            }
          },
          items: listEPs.map<DropdownMenuItem<String>>((EP value) {
            return DropdownMenuItem<String>(
              value: value.name,
              child: value.widget,
            );
          }).toList(),
        ),
      ],
    );
  }
}
