// This file is automatically generated, so please do not edit it.
// @generated by `flutter_rust_bridge`@ 2.7.0.

// ignore_for_file: invalid_use_of_internal_member, unused_import, unnecessary_import

import '../frb_generated.dart';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';

// These functions are ignored because they are not marked as `pub`: `import_model`, `intersection`, `iou`, `prepare_input`, `process_output`, `run_model`, `union`

Future<void> setModel({required String value}) =>
    RustLib.instance.api.crateApiInferenceSetModel(value: value);

Future<String> detect({required String filePath}) =>
    RustLib.instance.api.crateApiInferenceDetect(filePath: filePath);
