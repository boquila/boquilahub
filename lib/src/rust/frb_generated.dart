// This file is automatically generated, so please do not edit it.
// @generated by `flutter_rust_bridge`@ 2.7.0.

// ignore_for_file: unused_import, unused_element, unnecessary_import, duplicate_ignore, invalid_use_of_internal_member, annotate_overrides, non_constant_identifier_names, curly_braces_in_flow_control_structures, prefer_const_literals_to_create_immutables, unused_field

import 'api/abstractions.dart';
import 'api/inference.dart';
import 'api/utils.dart';
import 'dart:async';
import 'dart:convert';
import 'frb_generated.dart';
import 'frb_generated.io.dart'
    if (dart.library.js_interop) 'frb_generated.web.dart';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';

/// Main entrypoint of the Rust API
class RustLib extends BaseEntrypoint<RustLibApi, RustLibApiImpl, RustLibWire> {
  @internal
  static final instance = RustLib._();

  RustLib._();

  /// Initialize flutter_rust_bridge
  static Future<void> init({
    RustLibApi? api,
    BaseHandler? handler,
    ExternalLibrary? externalLibrary,
  }) async {
    await instance.initImpl(
      api: api,
      handler: handler,
      externalLibrary: externalLibrary,
    );
  }

  /// Initialize flutter_rust_bridge in mock mode.
  /// No libraries for FFI are loaded.
  static void initMock({
    required RustLibApi api,
  }) {
    instance.initMockImpl(
      api: api,
    );
  }

  /// Dispose flutter_rust_bridge
  ///
  /// The call to this function is optional, since flutter_rust_bridge (and everything else)
  /// is automatically disposed when the app stops.
  static void dispose() => instance.disposeImpl();

  @override
  ApiImplConstructor<RustLibApiImpl, RustLibWire> get apiImplConstructor =>
      RustLibApiImpl.new;

  @override
  WireConstructor<RustLibWire> get wireConstructor =>
      RustLibWire.fromExternalLibrary;

  @override
  Future<void> executeRustInitializers() async {
    await api.crateApiInferenceInitApp();
  }

  @override
  ExternalLibraryLoaderConfig get defaultExternalLibraryLoaderConfig =>
      kDefaultExternalLibraryLoaderConfig;

  @override
  String get codegenVersion => '2.7.0';

  @override
  int get rustContentHash => -1298845494;

  static const kDefaultExternalLibraryLoaderConfig =
      ExternalLibraryLoaderConfig(
    stem: 'rust_lib_boquilahub',
    ioDirectory: 'rust/target/release/',
    webPrefix: 'pkg/',
  );
}

abstract class RustLibApi extends BaseApi {
  Future<String> crateApiInferenceDetect({required String filePath});

  Future<double> crateApiUtilsGetCudaVersion();

  Future<void> crateApiInferenceInitApp();

  Future<void> crateApiInferenceSetModel({required String value});

  Future<double> crateApiAbstractionsXywHnArea({required XYWHn that});

  Future<XYWHn> crateApiAbstractionsXywHnNew(
      {required double x,
      required double y,
      required double w,
      required double h,
      required BigInt classId,
      required double probability});

  Future<XYXYn> crateApiAbstractionsXywHnToxyxy({required XYWHn that});

  Future<double> crateApiAbstractionsXyxYnArea({required XYXYn that});

  Future<double> crateApiAbstractionsXyxYnIntersect(
      {required XYXYn that, required XYXYn other});

  Future<double> crateApiAbstractionsXyxYnIou(
      {required XYXYn that, required XYXYn other});

  Future<XYXYn> crateApiAbstractionsXyxYnNew(
      {required double x1,
      required double y1,
      required double x2,
      required double y2,
      required BigInt classId,
      required double probability});

  Future<XYWHn> crateApiAbstractionsXyxYnToxywh({required XYXYn that});
}

class RustLibApiImpl extends RustLibApiImplPlatform implements RustLibApi {
  RustLibApiImpl({
    required super.handler,
    required super.wire,
    required super.generalizedFrbRustBinding,
    required super.portManager,
  });

  @override
  Future<String> crateApiInferenceDetect({required String filePath}) {
    return handler.executeNormal(NormalTask(
      callFfi: (port_) {
        final serializer = SseSerializer(generalizedFrbRustBinding);
        sse_encode_String(filePath, serializer);
        pdeCallFfi(generalizedFrbRustBinding, serializer,
            funcId: 1, port: port_);
      },
      codec: SseCodec(
        decodeSuccessData: sse_decode_String,
        decodeErrorData: null,
      ),
      constMeta: kCrateApiInferenceDetectConstMeta,
      argValues: [filePath],
      apiImpl: this,
    ));
  }

  TaskConstMeta get kCrateApiInferenceDetectConstMeta => const TaskConstMeta(
        debugName: "detect",
        argNames: ["filePath"],
      );

  @override
  Future<double> crateApiUtilsGetCudaVersion() {
    return handler.executeNormal(NormalTask(
      callFfi: (port_) {
        final serializer = SseSerializer(generalizedFrbRustBinding);
        pdeCallFfi(generalizedFrbRustBinding, serializer,
            funcId: 2, port: port_);
      },
      codec: SseCodec(
        decodeSuccessData: sse_decode_f_64,
        decodeErrorData: null,
      ),
      constMeta: kCrateApiUtilsGetCudaVersionConstMeta,
      argValues: [],
      apiImpl: this,
    ));
  }

  TaskConstMeta get kCrateApiUtilsGetCudaVersionConstMeta =>
      const TaskConstMeta(
        debugName: "get_cuda_version",
        argNames: [],
      );

  @override
  Future<void> crateApiInferenceInitApp() {
    return handler.executeNormal(NormalTask(
      callFfi: (port_) {
        final serializer = SseSerializer(generalizedFrbRustBinding);
        pdeCallFfi(generalizedFrbRustBinding, serializer,
            funcId: 3, port: port_);
      },
      codec: SseCodec(
        decodeSuccessData: sse_decode_unit,
        decodeErrorData: null,
      ),
      constMeta: kCrateApiInferenceInitAppConstMeta,
      argValues: [],
      apiImpl: this,
    ));
  }

  TaskConstMeta get kCrateApiInferenceInitAppConstMeta => const TaskConstMeta(
        debugName: "init_app",
        argNames: [],
      );

  @override
  Future<void> crateApiInferenceSetModel({required String value}) {
    return handler.executeNormal(NormalTask(
      callFfi: (port_) {
        final serializer = SseSerializer(generalizedFrbRustBinding);
        sse_encode_String(value, serializer);
        pdeCallFfi(generalizedFrbRustBinding, serializer,
            funcId: 4, port: port_);
      },
      codec: SseCodec(
        decodeSuccessData: sse_decode_unit,
        decodeErrorData: null,
      ),
      constMeta: kCrateApiInferenceSetModelConstMeta,
      argValues: [value],
      apiImpl: this,
    ));
  }

  TaskConstMeta get kCrateApiInferenceSetModelConstMeta => const TaskConstMeta(
        debugName: "set_model",
        argNames: ["value"],
      );

  @override
  Future<double> crateApiAbstractionsXywHnArea({required XYWHn that}) {
    return handler.executeNormal(NormalTask(
      callFfi: (port_) {
        final serializer = SseSerializer(generalizedFrbRustBinding);
        sse_encode_box_autoadd_xyw_hn(that, serializer);
        pdeCallFfi(generalizedFrbRustBinding, serializer,
            funcId: 5, port: port_);
      },
      codec: SseCodec(
        decodeSuccessData: sse_decode_f_32,
        decodeErrorData: null,
      ),
      constMeta: kCrateApiAbstractionsXywHnAreaConstMeta,
      argValues: [that],
      apiImpl: this,
    ));
  }

  TaskConstMeta get kCrateApiAbstractionsXywHnAreaConstMeta =>
      const TaskConstMeta(
        debugName: "xyw_hn_area",
        argNames: ["that"],
      );

  @override
  Future<XYWHn> crateApiAbstractionsXywHnNew(
      {required double x,
      required double y,
      required double w,
      required double h,
      required BigInt classId,
      required double probability}) {
    return handler.executeNormal(NormalTask(
      callFfi: (port_) {
        final serializer = SseSerializer(generalizedFrbRustBinding);
        sse_encode_f_32(x, serializer);
        sse_encode_f_32(y, serializer);
        sse_encode_f_32(w, serializer);
        sse_encode_f_32(h, serializer);
        sse_encode_usize(classId, serializer);
        sse_encode_f_32(probability, serializer);
        pdeCallFfi(generalizedFrbRustBinding, serializer,
            funcId: 6, port: port_);
      },
      codec: SseCodec(
        decodeSuccessData: sse_decode_xyw_hn,
        decodeErrorData: null,
      ),
      constMeta: kCrateApiAbstractionsXywHnNewConstMeta,
      argValues: [x, y, w, h, classId, probability],
      apiImpl: this,
    ));
  }

  TaskConstMeta get kCrateApiAbstractionsXywHnNewConstMeta =>
      const TaskConstMeta(
        debugName: "xyw_hn_new",
        argNames: ["x", "y", "w", "h", "classId", "probability"],
      );

  @override
  Future<XYXYn> crateApiAbstractionsXywHnToxyxy({required XYWHn that}) {
    return handler.executeNormal(NormalTask(
      callFfi: (port_) {
        final serializer = SseSerializer(generalizedFrbRustBinding);
        sse_encode_box_autoadd_xyw_hn(that, serializer);
        pdeCallFfi(generalizedFrbRustBinding, serializer,
            funcId: 7, port: port_);
      },
      codec: SseCodec(
        decodeSuccessData: sse_decode_xyx_yn,
        decodeErrorData: null,
      ),
      constMeta: kCrateApiAbstractionsXywHnToxyxyConstMeta,
      argValues: [that],
      apiImpl: this,
    ));
  }

  TaskConstMeta get kCrateApiAbstractionsXywHnToxyxyConstMeta =>
      const TaskConstMeta(
        debugName: "xyw_hn_toxyxy",
        argNames: ["that"],
      );

  @override
  Future<double> crateApiAbstractionsXyxYnArea({required XYXYn that}) {
    return handler.executeNormal(NormalTask(
      callFfi: (port_) {
        final serializer = SseSerializer(generalizedFrbRustBinding);
        sse_encode_box_autoadd_xyx_yn(that, serializer);
        pdeCallFfi(generalizedFrbRustBinding, serializer,
            funcId: 8, port: port_);
      },
      codec: SseCodec(
        decodeSuccessData: sse_decode_f_32,
        decodeErrorData: null,
      ),
      constMeta: kCrateApiAbstractionsXyxYnAreaConstMeta,
      argValues: [that],
      apiImpl: this,
    ));
  }

  TaskConstMeta get kCrateApiAbstractionsXyxYnAreaConstMeta =>
      const TaskConstMeta(
        debugName: "xyx_yn_area",
        argNames: ["that"],
      );

  @override
  Future<double> crateApiAbstractionsXyxYnIntersect(
      {required XYXYn that, required XYXYn other}) {
    return handler.executeNormal(NormalTask(
      callFfi: (port_) {
        final serializer = SseSerializer(generalizedFrbRustBinding);
        sse_encode_box_autoadd_xyx_yn(that, serializer);
        sse_encode_box_autoadd_xyx_yn(other, serializer);
        pdeCallFfi(generalizedFrbRustBinding, serializer,
            funcId: 9, port: port_);
      },
      codec: SseCodec(
        decodeSuccessData: sse_decode_f_32,
        decodeErrorData: null,
      ),
      constMeta: kCrateApiAbstractionsXyxYnIntersectConstMeta,
      argValues: [that, other],
      apiImpl: this,
    ));
  }

  TaskConstMeta get kCrateApiAbstractionsXyxYnIntersectConstMeta =>
      const TaskConstMeta(
        debugName: "xyx_yn_intersect",
        argNames: ["that", "other"],
      );

  @override
  Future<double> crateApiAbstractionsXyxYnIou(
      {required XYXYn that, required XYXYn other}) {
    return handler.executeNormal(NormalTask(
      callFfi: (port_) {
        final serializer = SseSerializer(generalizedFrbRustBinding);
        sse_encode_box_autoadd_xyx_yn(that, serializer);
        sse_encode_box_autoadd_xyx_yn(other, serializer);
        pdeCallFfi(generalizedFrbRustBinding, serializer,
            funcId: 10, port: port_);
      },
      codec: SseCodec(
        decodeSuccessData: sse_decode_f_32,
        decodeErrorData: null,
      ),
      constMeta: kCrateApiAbstractionsXyxYnIouConstMeta,
      argValues: [that, other],
      apiImpl: this,
    ));
  }

  TaskConstMeta get kCrateApiAbstractionsXyxYnIouConstMeta =>
      const TaskConstMeta(
        debugName: "xyx_yn_iou",
        argNames: ["that", "other"],
      );

  @override
  Future<XYXYn> crateApiAbstractionsXyxYnNew(
      {required double x1,
      required double y1,
      required double x2,
      required double y2,
      required BigInt classId,
      required double probability}) {
    return handler.executeNormal(NormalTask(
      callFfi: (port_) {
        final serializer = SseSerializer(generalizedFrbRustBinding);
        sse_encode_f_32(x1, serializer);
        sse_encode_f_32(y1, serializer);
        sse_encode_f_32(x2, serializer);
        sse_encode_f_32(y2, serializer);
        sse_encode_usize(classId, serializer);
        sse_encode_f_32(probability, serializer);
        pdeCallFfi(generalizedFrbRustBinding, serializer,
            funcId: 11, port: port_);
      },
      codec: SseCodec(
        decodeSuccessData: sse_decode_xyx_yn,
        decodeErrorData: null,
      ),
      constMeta: kCrateApiAbstractionsXyxYnNewConstMeta,
      argValues: [x1, y1, x2, y2, classId, probability],
      apiImpl: this,
    ));
  }

  TaskConstMeta get kCrateApiAbstractionsXyxYnNewConstMeta =>
      const TaskConstMeta(
        debugName: "xyx_yn_new",
        argNames: ["x1", "y1", "x2", "y2", "classId", "probability"],
      );

  @override
  Future<XYWHn> crateApiAbstractionsXyxYnToxywh({required XYXYn that}) {
    return handler.executeNormal(NormalTask(
      callFfi: (port_) {
        final serializer = SseSerializer(generalizedFrbRustBinding);
        sse_encode_box_autoadd_xyx_yn(that, serializer);
        pdeCallFfi(generalizedFrbRustBinding, serializer,
            funcId: 12, port: port_);
      },
      codec: SseCodec(
        decodeSuccessData: sse_decode_xyw_hn,
        decodeErrorData: null,
      ),
      constMeta: kCrateApiAbstractionsXyxYnToxywhConstMeta,
      argValues: [that],
      apiImpl: this,
    ));
  }

  TaskConstMeta get kCrateApiAbstractionsXyxYnToxywhConstMeta =>
      const TaskConstMeta(
        debugName: "xyx_yn_toxywh",
        argNames: ["that"],
      );

  @protected
  String dco_decode_String(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    return raw as String;
  }

  @protected
  XYWHn dco_decode_box_autoadd_xyw_hn(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    return dco_decode_xyw_hn(raw);
  }

  @protected
  XYXYn dco_decode_box_autoadd_xyx_yn(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    return dco_decode_xyx_yn(raw);
  }

  @protected
  double dco_decode_f_32(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    return raw as double;
  }

  @protected
  double dco_decode_f_64(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    return raw as double;
  }

  @protected
  Uint8List dco_decode_list_prim_u_8_strict(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    return raw as Uint8List;
  }

  @protected
  int dco_decode_u_8(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    return raw as int;
  }

  @protected
  void dco_decode_unit(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    return;
  }

  @protected
  BigInt dco_decode_usize(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    return dcoDecodeU64(raw);
  }

  @protected
  XYWHn dco_decode_xyw_hn(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    final arr = raw as List<dynamic>;
    if (arr.length != 6)
      throw Exception('unexpected arr length: expect 6 but see ${arr.length}');
    return XYWHn(
      x: dco_decode_f_32(arr[0]),
      y: dco_decode_f_32(arr[1]),
      w: dco_decode_f_32(arr[2]),
      h: dco_decode_f_32(arr[3]),
      classId: dco_decode_usize(arr[4]),
      probability: dco_decode_f_32(arr[5]),
    );
  }

  @protected
  XYXYn dco_decode_xyx_yn(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    final arr = raw as List<dynamic>;
    if (arr.length != 6)
      throw Exception('unexpected arr length: expect 6 but see ${arr.length}');
    return XYXYn(
      x1: dco_decode_f_32(arr[0]),
      y1: dco_decode_f_32(arr[1]),
      x2: dco_decode_f_32(arr[2]),
      y2: dco_decode_f_32(arr[3]),
      classId: dco_decode_usize(arr[4]),
      probability: dco_decode_f_32(arr[5]),
    );
  }

  @protected
  String sse_decode_String(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    var inner = sse_decode_list_prim_u_8_strict(deserializer);
    return utf8.decoder.convert(inner);
  }

  @protected
  XYWHn sse_decode_box_autoadd_xyw_hn(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    return (sse_decode_xyw_hn(deserializer));
  }

  @protected
  XYXYn sse_decode_box_autoadd_xyx_yn(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    return (sse_decode_xyx_yn(deserializer));
  }

  @protected
  double sse_decode_f_32(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    return deserializer.buffer.getFloat32();
  }

  @protected
  double sse_decode_f_64(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    return deserializer.buffer.getFloat64();
  }

  @protected
  Uint8List sse_decode_list_prim_u_8_strict(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    var len_ = sse_decode_i_32(deserializer);
    return deserializer.buffer.getUint8List(len_);
  }

  @protected
  int sse_decode_u_8(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    return deserializer.buffer.getUint8();
  }

  @protected
  void sse_decode_unit(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
  }

  @protected
  BigInt sse_decode_usize(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    return deserializer.buffer.getBigUint64();
  }

  @protected
  XYWHn sse_decode_xyw_hn(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    var var_x = sse_decode_f_32(deserializer);
    var var_y = sse_decode_f_32(deserializer);
    var var_w = sse_decode_f_32(deserializer);
    var var_h = sse_decode_f_32(deserializer);
    var var_classId = sse_decode_usize(deserializer);
    var var_probability = sse_decode_f_32(deserializer);
    return XYWHn(
        x: var_x,
        y: var_y,
        w: var_w,
        h: var_h,
        classId: var_classId,
        probability: var_probability);
  }

  @protected
  XYXYn sse_decode_xyx_yn(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    var var_x1 = sse_decode_f_32(deserializer);
    var var_y1 = sse_decode_f_32(deserializer);
    var var_x2 = sse_decode_f_32(deserializer);
    var var_y2 = sse_decode_f_32(deserializer);
    var var_classId = sse_decode_usize(deserializer);
    var var_probability = sse_decode_f_32(deserializer);
    return XYXYn(
        x1: var_x1,
        y1: var_y1,
        x2: var_x2,
        y2: var_y2,
        classId: var_classId,
        probability: var_probability);
  }

  @protected
  int sse_decode_i_32(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    return deserializer.buffer.getInt32();
  }

  @protected
  bool sse_decode_bool(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    return deserializer.buffer.getUint8() != 0;
  }

  @protected
  void sse_encode_String(String self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    sse_encode_list_prim_u_8_strict(utf8.encoder.convert(self), serializer);
  }

  @protected
  void sse_encode_box_autoadd_xyw_hn(XYWHn self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    sse_encode_xyw_hn(self, serializer);
  }

  @protected
  void sse_encode_box_autoadd_xyx_yn(XYXYn self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    sse_encode_xyx_yn(self, serializer);
  }

  @protected
  void sse_encode_f_32(double self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    serializer.buffer.putFloat32(self);
  }

  @protected
  void sse_encode_f_64(double self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    serializer.buffer.putFloat64(self);
  }

  @protected
  void sse_encode_list_prim_u_8_strict(
      Uint8List self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    sse_encode_i_32(self.length, serializer);
    serializer.buffer.putUint8List(self);
  }

  @protected
  void sse_encode_u_8(int self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    serializer.buffer.putUint8(self);
  }

  @protected
  void sse_encode_unit(void self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
  }

  @protected
  void sse_encode_usize(BigInt self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    serializer.buffer.putBigUint64(self);
  }

  @protected
  void sse_encode_xyw_hn(XYWHn self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    sse_encode_f_32(self.x, serializer);
    sse_encode_f_32(self.y, serializer);
    sse_encode_f_32(self.w, serializer);
    sse_encode_f_32(self.h, serializer);
    sse_encode_usize(self.classId, serializer);
    sse_encode_f_32(self.probability, serializer);
  }

  @protected
  void sse_encode_xyx_yn(XYXYn self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    sse_encode_f_32(self.x1, serializer);
    sse_encode_f_32(self.y1, serializer);
    sse_encode_f_32(self.x2, serializer);
    sse_encode_f_32(self.y2, serializer);
    sse_encode_usize(self.classId, serializer);
    sse_encode_f_32(self.probability, serializer);
  }

  @protected
  void sse_encode_i_32(int self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    serializer.buffer.putInt32(self);
  }

  @protected
  void sse_encode_bool(bool self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    serializer.buffer.putUint8(self ? 1 : 0);
  }
}
