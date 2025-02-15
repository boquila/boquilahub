use ndarray::{s, Array, Axis, IxDyn};
use super::abstractions::{nms, BoundingBoxTrait, XYXY};

// Function used to convert the output tensor from YOLO to an Vec<XYXY>
pub fn process_output(
    output: &Array<f32, IxDyn>,
    img_width: u32,
    img_height: u32,
    input_width: u32, 
    input_height: u32
) -> Vec<XYXY> {
    let mut boxes = Vec::new();
    let output = output.slice(s![.., .., 0]);
    for row in output.axis_iter(Axis(0)) {
        let row: Vec<_> = row.iter().map(|x| *x).collect();
        let (class_id, prob) = row
            .iter()
            .skip(4)
            .enumerate()
            .map(|(index, value)| (index, *value))
            .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
            .unwrap();
        if prob < 0.45 {
            continue;
        }
        let label = class_id as u16;
        let xc = row[0] / input_width as f32 * (img_width as f32);
        let yc = row[1] / input_height as f32 * (img_height as f32);
        let w = row[2] / input_width as f32 * (img_width as f32);
        let h = row[3] / input_height as f32 * (img_height as f32);
        let x1 = xc - w / 2.0;
        let x2 = xc + w / 2.0;
        let y1 = yc - h / 2.0;
        let y2 = yc + h / 2.0;
        let temp = XYXY::new(x1,y1,x2,y2,prob,label,);
        boxes.push(temp);
    }

    let result = nms(boxes,0.7);
    return result;
}