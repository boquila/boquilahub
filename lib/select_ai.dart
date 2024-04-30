import 'package:flutter/material.dart';
import 'src/resources/objects.dart';

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

  @override
  Widget build(BuildContext context) {
    return DropdownButton<String>(
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
      },
      items: listAIs.map<DropdownMenuItem<String>>((AI value) {
        return DropdownMenuItem<String>(
          value: value.description,
          child: Text(value.description),
        );
      }).toList(),
    );
  }
}
