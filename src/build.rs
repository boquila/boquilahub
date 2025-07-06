extern crate embed_resource;

pub fn main() {
    embed_resource::compile("app.rc", embed_resource::NONE).manifest_optional().unwrap();
}