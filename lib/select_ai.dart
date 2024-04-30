import 'src/resources/palettes.dart';
import 'package:flutter/material.dart';

const List<String> list = <String>[
  '🖼️ Ánimales (genérico)',
  '🖼️ Ánimales (especies)',
  '🖼️ Hongos (especies)',
  '🔊 Hídrofonos',
  '🔊 Aves (especies)',
  '🎥 Incendios'
];

class DropdownButtonExample extends StatefulWidget {
  const DropdownButtonExample({super.key});

  @override
  State<DropdownButtonExample> createState() => _DropdownButtonExampleState();
}

class _DropdownButtonExampleState extends State<DropdownButtonExample> {
  String dropdownValue = list.first;

  @override
  Widget build(BuildContext context) {

    return DropdownButton<String>(
      value: dropdownValue,
      icon: const Icon(Icons.search),
      elevation: 4,
      style: TextStyle(color: currentcolors[4]),
      underline: Container(
        height: 0.25,
        width: 5,
        color: currentcolors[2],
      ),
      onChanged: (String? value) {
        setState(() {
          if (value == list.first) {
            setState(() {
              currentcolors = terra;
              dropdownValue = value!;
            });
          } else  {
            showDialog(
                context: context,
                builder: (context) => AlertDialog(actions: [
                      ElevatedButton(
                          onPressed: () {
                            Navigator.pop(context);
                          },
                          child: const Text("Ok"))
                    ], title: const Text("Sección no disponible")));
            dropdownValue = list.first;
          }
        });
      },
      items: list.map<DropdownMenuItem<String>>((String value) {
        return DropdownMenuItem<String>(
          value: value,
          child: Text(value),
        );
      }).toList(),
    );
  }
}
