extern crate embed_resource;

pub fn main() {
    embed_resource::compile("assets/app.rc", embed_resource::NONE).manifest_optional().unwrap();
}