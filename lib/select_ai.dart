import 'src/resources/palettes.dart';
import 'package:flutter/material.dart';
import 'src/resources/objects.dart';

// const List<String> list = <String>[
//   '🖼️ Ánimales (genérico)',
//   '🖼️ Ánimales (especies)',
//   '🖼️ Hongos (especies)',
//   '🔊 Hídrofonos',
//   '🔊 Aves (especies)',
//   '🎥 Incendios'
// ];

class SelectAIPage extends StatefulWidget {
  const SelectAIPage({super.key});

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
      style: TextStyle(color: currentcolors[4]),
      underline: Container(
        height: 0.25,
        width: 5,
        color: currentcolors[2],
      ),
      onChanged: (String? value) {
        setState(() {
          dropdownValue = value!;
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
