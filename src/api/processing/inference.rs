use crate::api::bq::ort_err;
use ndarray::{Array, Ix4};

pub fn inference<'a>(
    session: &'a ort::session::Session,
    input: &'a Array<f32, Ix4>,
    input_name: &str,
) -> anyhow::Result<ort::session::SessionOutputs<'a>> {
    let start = std::time::Instant::now();
    #[allow(mutable_transmutes)]
    let session: &'a mut ort::session::Session = unsafe { std::mem::transmute(session) };
    println!("Elapsed: {:?}", start.elapsed());
    let input = ort::value::TensorRef::from_array_view(input.view()).map_err(ort_err)?;
    Ok(session.run(ort::inputs![input_name => input]).map_err(ort_err)?)
}
