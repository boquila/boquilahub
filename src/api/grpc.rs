pub mod pb {
    tonic::include_proto!("boquila");
}

use super::abstractions::*;
use super::bq::process_imgbuf;

pub struct BoquilaHubService;

fn bitvec_to_bytes(bv: &bitvec::vec::BitVec) -> Vec<u8> {
    let len = bv.len();
    let byte_len = (len + 7) / 8;
    let mut bytes = vec![0u8; byte_len];
    for i in 0..len {
        if bv[i] {
            bytes[i / 8] |= 1 << (i % 8);
        }
    }
    bytes
}

fn xyxy_to_pb(xyxy: &XYXY) -> pb::BBox {
    pb::BBox {
        x1: xyxy.x1,
        y1: xyxy.y1,
        x2: xyxy.x2,
        y2: xyxy.y2,
        prob: xyxy.prob,
        class_id: xyxy.class_id,
    }
}

fn prob_space_to_pb(ps: &ProbSpace) -> pb::Probabilities {
    pb::Probabilities {
        classes: ps.classes.clone(),
        probs: ps.probs.clone(),
        class_ids: ps.classes_ids.clone(),
    }
}

fn xyxyc_to_pb(xyxyc: &XYXYc) -> pb::DetectedObject {
    pb::DetectedObject {
        bbox: Some(xyxy_to_pb(&xyxyc.xyxy)),
        label: xyxyc.label.clone(),
        extra_cls: xyxyc.extra_cls.as_ref().map(prob_space_to_pb),
    }
}

fn segc_to_pb(segc: &SEGc) -> pb::SegmentedObject {
    pb::SegmentedObject {
        mask: Some(pb::BitMatrix {
            data: bitvec_to_bytes(&segc.mask.data),
            width: segc.mask.width as u32,
            height: segc.mask.height as u32,
            num_bits: segc.mask.data.len() as u32,
        }),
        bbox: Some(xyxyc_to_pb(&segc.bbox)),
    }
}

fn audio_prob_to_pb(ap: &AudioProb) -> pb::AudioPrediction {
    pb::AudioPrediction {
        start: ap.start,
        end: ap.end,
        class_id: ap.class_id,
        prob: ap.prob,
        positive: ap.positive,
        label: ap.label.clone(),
    }
}

fn ai_outputs_to_response(output: AIOutputs) -> pb::DetectResponse {
    match output {
        AIOutputs::ObjectDetection(detections) => pb::DetectResponse {
            output: Some(pb::detect_response::Output::ObjectDetection(
                pb::ObjectDetectionResult {
                    detections: detections.iter().map(xyxyc_to_pb).collect(),
                },
            )),
        },
        AIOutputs::Classification(prob_space) => pb::DetectResponse {
            output: Some(pb::detect_response::Output::Classification(
                pb::ClassificationResult {
                    probabilities: Some(prob_space_to_pb(&prob_space)),
                },
            )),
        },
        AIOutputs::Segmentation(segments) => pb::DetectResponse {
            output: Some(pb::detect_response::Output::Segmentation(
                pb::SegmentationResult {
                    segments: segments.iter().map(segc_to_pb).collect(),
                },
            )),
        },
        AIOutputs::AudioClassification(probs) => pb::DetectResponse {
            output: Some(pb::detect_response::Output::AudioClassification(
                pb::AudioClassificationResult {
                    predictions: probs.iter().map(audio_prob_to_pb).collect(),
                },
            )),
        },
    }
}

fn pb_to_xyxy(bbox: &pb::BBox) -> XYXY {
    XYXY {
        x1: bbox.x1,
        y1: bbox.y1,
        x2: bbox.x2,
        y2: bbox.y2,
        prob: bbox.prob,
        class_id: bbox.class_id,
    }
}

fn pb_to_prob_space(ps: &pb::Probabilities) -> ProbSpace {
    ProbSpace::new(
        ps.classes.clone(),
        ps.probs.clone(),
        ps.class_ids.clone(),
    )
}

fn pb_to_xyxyc(obj: &pb::DetectedObject) -> XYXYc {
    let bbox = obj.bbox.as_ref().map(pb_to_xyxy).unwrap_or(XYXY {
        x1: 0.0,
        y1: 0.0,
        x2: 0.0,
        y2: 0.0,
        prob: 0.0,
        class_id: 0,
    });
    let mut xyxyc = XYXYc::new(bbox, obj.label.clone());
    if let Some(ref extra_cls) = obj.extra_cls {
        xyxyc.extra_cls = Some(pb_to_prob_space(extra_cls));
    }
    xyxyc
}

fn pb_to_bit_matrix(bm: &pb::BitMatrix) -> BitMatrix {
    let num_bits = bm.num_bits as usize;
    let mut bv = bitvec::vec::BitVec::with_capacity(num_bits);
    for i in 0..num_bits {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        let is_set = (bm.data.get(byte_idx).copied().unwrap_or(0) >> bit_idx) & 1 == 1;
        bv.push(is_set);
    }
    BitMatrix {
        data: bv,
        width: bm.width as usize,
        height: bm.height as usize,
    }
}

fn pb_to_audio_prob(ap: &pb::AudioPrediction) -> AudioProb {
    AudioProb {
        start: ap.start,
        end: ap.end,
        class_id: ap.class_id,
        prob: ap.prob,
        positive: ap.positive,
        label: ap.label.clone(),
    }
}

fn response_to_ai_outputs(resp: pb::DetectResponse) -> AIOutputs {
    match resp.output {
        Some(pb::detect_response::Output::ObjectDetection(od)) => {
            AIOutputs::ObjectDetection(od.detections.iter().map(pb_to_xyxyc).collect())
        }
        Some(pb::detect_response::Output::Classification(cls)) => {
            let ps = cls.probabilities.as_ref().map(pb_to_prob_space).unwrap_or_else(|| ProbSpace::new(vec![], vec![], vec![]));
            AIOutputs::Classification(ps)
        }
        Some(pb::detect_response::Output::Segmentation(seg)) => {
            AIOutputs::Segmentation(
                seg.segments.iter().map(|s| {
                    let mask = s.mask.as_ref().map(pb_to_bit_matrix).unwrap_or(BitMatrix {
                        data: bitvec::vec::BitVec::new(),
                        width: 0,
                        height: 0,
                    });
                    let bbox = pb_to_xyxyc(s.bbox.as_ref().unwrap_or(&pb::DetectedObject::default()));
                    SEGc::new(mask, bbox)
                }).collect(),
            )
        }
        Some(pb::detect_response::Output::AudioClassification(ac)) => {
            AIOutputs::AudioClassification(ac.predictions.iter().map(pb_to_audio_prob).collect())
        }
        None => AIOutputs::Classification(ProbSpace::new(vec![], vec![], vec![])),
    }
}

#[tonic::async_trait]
impl pb::boquila_hub_server::BoquilaHub for BoquilaHubService {
    async fn detect(
        &self,
        request: tonic::Request<pb::DetectRequest>,
    ) -> Result<tonic::Response<pb::DetectResponse>, tonic::Status> {
        let data = request.into_inner().image;

        let imgbuf = image::load_from_memory(&data)
            .map_err(|e| tonic::Status::invalid_argument(format!("Invalid image: {}", e)))?
            .into_rgb8();

        let result = process_imgbuf(&imgbuf);
        let response = ai_outputs_to_response(result);

        Ok(tonic::Response::new(response))
    }

    async fn health(
        &self,
        _request: tonic::Request<pb::HealthRequest>,
    ) -> Result<tonic::Response<pb::HealthResponse>, tonic::Status> {
        Ok(tonic::Response::new(pb::HealthResponse {
            message: "BoquilaHUB gRPC API!".to_string(),
        }))
    }
}

pub async fn run_grpc(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("0.0.0.0:{}", port).parse()?;
    let service = BoquilaHubService;
    let _ = addr;
    tonic::transport::Server::builder()
        .add_service(pb::boquila_hub_server::BoquilaHubServer::new(service))
        .serve(addr)
        .await?;
    Ok(())
}

pub async fn detect_remotely_grpc(
    url: String,
    buffer: Vec<u8>,
) -> Result<AIOutputs, Box<dyn std::error::Error + Send + Sync>> {
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        async {
            let mut client = pb::boquila_hub_client::BoquilaHubClient::connect(url).await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
            let request = tonic::Request::new(pb::DetectRequest { image: buffer });
            let response = client.detect(request).await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(response.into_inner())
        },
    )
    .await
    .map_err(|_| Box::new(std::io::Error::new(std::io::ErrorKind::TimedOut, "gRPC connection timed out")) as Box<dyn std::error::Error + Send + Sync>)??;
    let result = response_to_ai_outputs(result);
    Ok(result)
}

pub async fn check_boquila_hub_grpc(url: String) -> bool {
    let Ok(client) = tokio::time::timeout(
        std::time::Duration::from_secs(3),
        pb::boquila_hub_client::BoquilaHubClient::connect(url),
    )
    .await
    else {
        return false;
    };
    let Ok(mut client) = client else {
        return false;
    };

    let request = tonic::Request::new(pb::HealthRequest {});
    let Ok(response) = client.health(request).await else {
        return false;
    };

    response.into_inner().message == "BoquilaHUB gRPC API!"
}