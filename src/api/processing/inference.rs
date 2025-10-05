use ndarray::{Array, Ix4};

pub fn inference<'a>(
    session: &'a ort::session::Session,
    input: &'a Array<f32, Ix4>,
    input_name: &str,
) -> ort::session::SessionOutputs<'a, 'a> {
    return session
        .run(ort::inputs![input_name => input.view()].unwrap())
        .unwrap();
}
