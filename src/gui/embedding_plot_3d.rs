//! GPU point cloud and camera interaction for the embedding plot.

use eframe::{egui, egui_glow, glow};
use glow::HasContext as _;
use std::mem;
use std::sync::{Arc, Mutex};

#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 3],
    color: [u8; 4],
}

#[derive(Clone, Copy)]
struct Camera {
    yaw: f32,
    pitch: f32,
    zoom: f32,
    pan: egui::Vec2,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            yaw: 0.55,
            pitch: 0.5,
            zoom: 0.62,
            pan: egui::Vec2::ZERO,
        }
    }
}

impl Camera {
    fn rotate(&mut self, delta: egui::Vec2) {
        self.yaw = (self.yaw - delta.x * 0.008).rem_euclid(std::f32::consts::TAU);
        self.pitch = (self.pitch + delta.y * 0.008).clamp(-1.5, 1.5);
    }

    fn fitted(self, rect: egui::Rect) -> Self {
        let fit = (rect.width() / rect.height().max(1.0) / 0.85).min(1.0);
        Self {
            zoom: self.zoom * fit,
            ..self
        }
    }
}

struct Projection {
    sy: f32,
    cy: f32,
    sp: f32,
    cp: f32,
    zoom: f32,
    center: egui::Pos2,
    half_height: f32,
    depth_scale: f32,
}

impl Projection {
    fn new(camera: Camera, rect: egui::Rect, depth_scale: f32) -> Self {
        let (sy, cy) = camera.yaw.sin_cos();
        let (sp, cp) = camera.pitch.sin_cos();
        Self {
            sy,
            cy,
            sp,
            cp,
            zoom: camera.zoom,
            center: rect.center() + camera.pan,
            half_height: rect.height() * 0.5,
            depth_scale,
        }
    }

    fn screen_position(&self, position: [f32; 3]) -> Option<(egui::Pos2, f32)> {
        let (sy, cy, sp, cp) = (self.sy, self.cy, self.sp, self.cp);
        let x = position[0] * cy - position[2] * self.depth_scale * sy;
        let z = position[0] * sy + position[2] * self.depth_scale * cy;
        let y = position[1] * cp - z * sp;
        let z = position[1] * sp + z * cp;
        let depth = 3.5 - z;
        if depth <= 0.1 {
            return None;
        }
        let scale = 2.4 * self.zoom / depth * self.half_height;
        let screen = self.center + egui::vec2(x * scale, -y * scale);
        Some((screen, depth))
    }
}

pub(super) struct Cloud3D {
    positions: Vec<[f32; 3]>,
    vertices: Arc<Vec<Vertex>>,
    camera: Camera,
    depth_scale: f32,
    floor_count: usize,
    frame_count: usize,
    intro_remaining: f32,
}

#[derive(Clone, Copy)]
pub(super) struct ViewOptions {
    pub grid: bool,
    pub frame: bool,
    pub expand_depth: bool,
}

impl Default for ViewOptions {
    fn default() -> Self {
        Self {
            grid: true,
            frame: true,
            expand_depth: true,
        }
    }
}

#[derive(Clone, Copy)]
struct DrawOptions {
    view: ViewOptions,
    point_count: usize,
    floor_count: usize,
    frame_count: usize,
    depth_scale: f32,
    clustered: bool,
}

const GUIDE_EXTENT: f32 = 1.08;

fn guide_line(vertices: &mut Vec<Vertex>, start: [f32; 3], end: [f32; 3]) {
    for position in [start, end] {
        vertices.push(Vertex {
            position,
            color: [255; 4],
        });
    }
}

pub(super) struct Interaction {
    pub hovered: Option<usize>,
    /// `Some(None)` means a click on empty canvas.
    pub clicked: Option<Option<usize>>,
}

impl Cloud3D {
    pub fn new(mut positions: Vec<[f32; 3]>) -> Self {
        let mut depth_scale = 1.0;
        if !positions.is_empty() {
            let mut low = [f32::INFINITY; 3];
            let mut high = [f32::NEG_INFINITY; 3];
            for point in &positions {
                for axis in 0..3 {
                    low[axis] = low[axis].min(point[axis]);
                    high[axis] = high[axis].max(point[axis]);
                }
            }
            let center = [0, 1, 2].map(|axis| (low[axis] + high[axis]) * 0.5);
            let radius = (0..3)
                .map(|axis| (high[axis] - low[axis]) * 0.5)
                .fold(0.0_f32, f32::max)
                .max(1e-9);
            let depth_radius = (high[2] - low[2]) * 0.5;
            if depth_radius > radius * 1e-4 {
                depth_scale = (radius / depth_radius).clamp(1.0, 4.0);
            }
            for point in &mut positions {
                for axis in 0..3 {
                    point[axis] = (point[axis] - center[axis]) / radius;
                }
            }
        }
        let mut vertices: Vec<Vertex> = positions
            .iter()
            .map(|&position| Vertex {
                position,
                color: [180, 190, 210, 255],
            })
            .collect();
        let e = GUIDE_EXTENT;
        for step in -2..=2 {
            let v = step as f32 * e * 0.5;
            guide_line(&mut vertices, [-e, -e, v], [e, -e, v]);
            guide_line(&mut vertices, [v, -e, -e], [v, -e, e]);
        }
        let floor_count = vertices.len() - positions.len();
        for &y in &[-e, e] {
            for &z in &[-e, e] {
                guide_line(&mut vertices, [-e, y, z], [e, y, z]);
            }
            for &x in &[-e, e] {
                guide_line(&mut vertices, [x, y, -e], [x, y, e]);
            }
        }
        for &x in &[-e, e] {
            for &z in &[-e, e] {
                guide_line(&mut vertices, [x, -e, z], [x, e, z]);
            }
        }
        let frame_count = vertices.len() - positions.len() - floor_count;
        Self {
            positions,
            vertices: Arc::new(vertices),
            camera: Camera::default(),
            depth_scale,
            floor_count,
            frame_count,
            intro_remaining: 1.4,
        }
    }

    pub fn set_clusters(&mut self, ids: &[usize], count: usize) {
        for (vertex, &id) in Arc::make_mut(&mut self.vertices).iter_mut().zip(ids) {
            vertex.color = super::embedding_plot::cluster_color(id, count).to_array();
        }
    }

    #[cfg(test)]
    pub fn positions(&self) -> &[[f32; 3]] {
        &self.positions
    }

    pub fn reset_camera(&mut self) {
        self.camera = Camera::default();
        self.intro_remaining = 0.0;
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        size: egui::Vec2,
        renderer: &Arc<Mutex<CloudRenderer>>,
        selected: Option<usize>,
        clustered: bool,
        view: ViewOptions,
    ) -> Interaction {
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
        let mut changed = false;
        if response.double_clicked() {
            self.reset_camera();
            changed = true;
        } else if response.dragged() {
            self.intro_remaining = 0.0;
            let delta = response.drag_delta();
            if ui.input(|input| input.modifiers.shift)
                || response.dragged_by(egui::PointerButton::Secondary)
            {
                self.camera.pan += delta;
            } else {
                self.camera.rotate(delta);
            }
            changed = true;
        }
        if response.hovered() {
            let scroll = ui.input(|input| input.smooth_scroll_delta.y);
            if scroll != 0.0 {
                self.intro_remaining = 0.0;
                self.camera.zoom = (self.camera.zoom * (scroll * 0.0015).exp()).clamp(0.4, 12.0);
                changed = true;
            }
        }
        if response.hovered() {
            self.intro_remaining = 0.0;
        } else if self.intro_remaining > 0.0 {
            let dt = ui.input(|input| input.stable_dt).min(0.05);
            self.camera.yaw = (self.camera.yaw + dt * 0.38).rem_euclid(std::f32::consts::TAU);
            self.intro_remaining = (self.intro_remaining - dt).max(0.0);
            changed = true;
        }
        if changed {
            ui.ctx().request_repaint();
        }

        let depth_scale = if view.expand_depth {
            self.depth_scale
        } else {
            1.0
        };
        let camera = self.camera.fitted(rect);
        let projection = Projection::new(camera, rect, depth_scale);
        let hovered = response.hover_pos().and_then(|pointer| {
            let mut best: Option<(usize, f32, f32)> = None;
            for (index, &point) in self.positions.iter().enumerate() {
                let Some((screen, depth)) = projection.screen_position(point) else {
                    continue;
                };
                let distance = screen.distance_sq(pointer);
                // Same circular footprint as the point sprite in the vertex shader.
                let radius = (14.0 / depth).clamp(2.0, 9.0);
                if distance > radius * radius {
                    continue;
                }
                if best.is_none_or(|(_, nearest, front)| {
                    depth < front - 1e-4 || ((depth - front).abs() <= 1e-4 && distance < nearest)
                }) {
                    best = Some((index, distance, depth));
                }
            }
            best.map(|(index, _, _)| index)
        });
        if hovered.is_some() {
            response
                .clone()
                .on_hover_cursor(egui::CursorIcon::PointingHand);
        } else if response.dragged() {
            response.clone().on_hover_cursor(egui::CursorIcon::Grabbing);
        } else {
            response.clone().on_hover_cursor(egui::CursorIcon::Grab);
        }
        let clicked = response.clicked().then_some(hovered);

        let dark = ui.visuals().dark_mode;
        let plain = if dark {
            [0.67, 0.76, 0.9, 1.0]
        } else {
            [0.16, 0.36, 0.58, 1.0]
        };
        let background = if dark {
            [0.055, 0.063, 0.078, 1.0]
        } else {
            [0.97, 0.975, 0.985, 1.0]
        };
        let vertices = self.vertices.clone();
        let callback_renderer = renderer.clone();
        let draw = DrawOptions {
            view,
            point_count: self.positions.len(),
            floor_count: self.floor_count,
            frame_count: self.frame_count,
            depth_scale,
            clustered,
        };
        let ctx = ui.ctx().clone();
        ui.painter().add(egui::PaintCallback {
            rect,
            callback: Arc::new(egui_glow::CallbackFn::new(move |info, painter| {
                let mut renderer = callback_renderer.lock().unwrap();
                match renderer.paint(
                    painter.gl(),
                    painter.intermediate_fbo(),
                    info,
                    &vertices,
                    camera,
                    draw,
                    plain,
                    background,
                ) {
                    Ok(()) => {
                        if renderer.error.take().is_some() {
                            ctx.request_repaint();
                        }
                    }
                    Err(error) => {
                        if renderer.error.as_deref() != Some(error.as_str()) {
                            renderer.error = Some(error);
                            ctx.request_repaint();
                        }
                    }
                }
            })),
        });

        let painter = ui.painter_at(rect);
        if let Some(index) = hovered.or(selected).filter(|_| view.grid) {
            let point = self.positions[index];
            if let (Some((top, _)), Some((floor, _))) = (
                projection.screen_position(point),
                projection.screen_position([point[0], -GUIDE_EXTENT, point[2]]),
            ) {
                let color = ui.visuals().selection.stroke.color.gamma_multiply(0.55);
                painter.line_segment([top, floor], egui::Stroke::new(1.0, color));
                painter.circle_filled(floor, 3.0, color);
            }
        }
        for (axis, label, color) in [
            ([1.0, 0.0, 0.0], "PC1", egui::Color32::from_rgb(225, 95, 95)),
            (
                [0.0, 1.0, 0.0],
                "PC2",
                egui::Color32::from_rgb(90, 195, 115),
            ),
            (
                [0.0, 0.0, 1.0],
                "PC3",
                egui::Color32::from_rgb(95, 145, 235),
            ),
        ] {
            let axis = [axis[0], axis[1], axis[2] / depth_scale];
            if let (Some((origin, _)), Some((end, _))) = (
                projection.screen_position([0.0; 3]),
                projection.screen_position(axis),
            ) {
                painter.line_segment(
                    [origin, end],
                    egui::Stroke::new(1.0, color.gamma_multiply(0.55)),
                );
                painter.text(
                    end,
                    egui::Align2::CENTER_CENTER,
                    label,
                    egui::FontId::proportional(11.0),
                    color,
                );
            }
        }
        if let Some(index) = selected {
            // Keep the selected location findable even when other points cover it.
            if let Some((screen, _)) = projection.screen_position(self.positions[index]) {
                painter.circle_stroke(
                    screen,
                    10.0,
                    egui::Stroke::new(2.0, ui.visuals().selection.stroke.color),
                );
            }
        }
        if let Some(index) = hovered {
            if let Some((screen, _)) = projection.screen_position(self.positions[index]) {
                painter.circle_stroke(
                    screen,
                    12.0,
                    egui::Stroke::new(2.0, ui.visuals().selection.stroke.color),
                );
            }
        }
        if let Ok(renderer) = renderer.lock() {
            if let Some(error) = &renderer.error {
                painter.text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    error,
                    egui::FontId::proportional(13.0),
                    egui::Color32::RED,
                );
            }
        }
        Interaction { hovered, clicked }
    }
}

#[derive(Default)]
pub(super) struct CloudRenderer {
    program: Option<glow::Program>,
    vao: Option<glow::VertexArray>,
    vbo: Option<glow::Buffer>,
    framebuffer: Option<glow::Framebuffer>,
    color: Option<glow::Texture>,
    depth: Option<glow::Renderbuffer>,
    size: (i32, i32),
    uploaded: Option<Arc<Vec<Vertex>>>,
    error: Option<String>,
}

impl CloudRenderer {
    fn paint(
        &mut self,
        gl: &glow::Context,
        target: Option<glow::Framebuffer>,
        info: egui::PaintCallbackInfo,
        vertices: &Arc<Vec<Vertex>>,
        camera: Camera,
        draw: DrawOptions,
        plain: [f32; 4],
        background: [f32; 4],
    ) -> Result<(), String> {
        let viewport = info.viewport_in_pixels();
        let width = viewport.width_px;
        let height = viewport.height_px;
        if width <= 0 || height <= 0 {
            return Ok(());
        }
        // All OpenGL work is deferred to egui's paint callback, when the context is current.
        unsafe {
            if let Err(error) = self
                .ensure_initialized(gl)
                .and_then(|_| self.ensure_size(gl, width, height))
            {
                gl.bind_framebuffer(glow::FRAMEBUFFER, target);
                self.destroy(gl);
                return Err(error);
            }
            let program = self.program.unwrap();
            let vao = self.vao.unwrap();
            let vbo = self.vbo.unwrap();
            if self
                .uploaded
                .as_ref()
                .is_none_or(|uploaded| !Arc::ptr_eq(uploaded, vertices))
            {
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
                let bytes = std::slice::from_raw_parts(
                    vertices.as_ptr().cast::<u8>(),
                    vertices.len() * mem::size_of::<Vertex>(),
                );
                gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytes, glow::STATIC_DRAW);
                self.uploaded = Some(vertices.clone());
            }
            let scissor = gl.is_enabled(glow::SCISSOR_TEST);
            gl.disable(glow::SCISSOR_TEST);
            gl.bind_framebuffer(glow::FRAMEBUFFER, self.framebuffer);
            gl.viewport(0, 0, width, height);
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LEQUAL);
            gl.depth_mask(true);
            gl.disable(glow::BLEND);
            let desktop_gl = !gl.get_parameter_string(glow::VERSION).contains("OpenGL ES");
            if desktop_gl {
                gl.enable(glow::PROGRAM_POINT_SIZE);
            }
            gl.clear_color(background[0], background[1], background[2], background[3]);
            gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
            gl.use_program(Some(program));
            gl.bind_vertex_array(Some(vao));
            gl.uniform_2_f32(
                gl.get_uniform_location(program, "u_angles").as_ref(),
                camera.yaw,
                camera.pitch,
            );
            gl.uniform_1_f32(
                gl.get_uniform_location(program, "u_zoom").as_ref(),
                camera.zoom,
            );
            gl.uniform_1_f32(
                gl.get_uniform_location(program, "u_aspect").as_ref(),
                width as f32 / height as f32,
            );
            gl.uniform_1_f32(
                gl.get_uniform_location(program, "u_dpi").as_ref(),
                info.pixels_per_point,
            );
            gl.uniform_2_f32(
                gl.get_uniform_location(program, "u_pan").as_ref(),
                camera.pan.x * info.pixels_per_point * 2.0 / width as f32,
                -camera.pan.y * info.pixels_per_point * 2.0 / height as f32,
            );
            gl.uniform_1_i32(
                gl.get_uniform_location(program, "u_clustered").as_ref(),
                draw.clustered as i32,
            );
            gl.uniform_1_f32(
                gl.get_uniform_location(program, "u_depth_scale").as_ref(),
                draw.depth_scale,
            );
            gl.uniform_4_f32(
                gl.get_uniform_location(program, "u_background").as_ref(),
                background[0],
                background[1],
                background[2],
                background[3],
            );
            let guide_uniform = gl.get_uniform_location(program, "u_guide");
            gl.uniform_1_i32(guide_uniform.as_ref(), 1);
            let plain_uniform = gl.get_uniform_location(program, "u_plain");
            let dark = background[0] < 0.5;
            if draw.view.grid {
                let color = if dark {
                    [0.17, 0.22, 0.29]
                } else {
                    [0.83, 0.86, 0.9]
                };
                gl.uniform_4_f32(plain_uniform.as_ref(), color[0], color[1], color[2], 1.0);
                gl.draw_arrays(
                    glow::LINES,
                    draw.point_count as i32,
                    draw.floor_count as i32,
                );
            }
            if draw.view.frame {
                let color = if dark {
                    [0.28, 0.36, 0.46]
                } else {
                    [0.67, 0.72, 0.78]
                };
                gl.uniform_4_f32(plain_uniform.as_ref(), color[0], color[1], color[2], 1.0);
                gl.draw_arrays(
                    glow::LINES,
                    (draw.point_count + draw.floor_count) as i32,
                    draw.frame_count as i32,
                );
            }
            gl.uniform_1_i32(guide_uniform.as_ref(), 0);
            gl.uniform_4_f32(
                plain_uniform.as_ref(),
                plain[0],
                plain[1],
                plain[2],
                plain[3],
            );
            gl.draw_arrays(
                glow::POINTS,
                0,
                draw.point_count.min(i32::MAX as usize) as i32,
            );
            gl.bind_vertex_array(None);
            gl.use_program(None);
            gl.disable(glow::DEPTH_TEST);
            if desktop_gl {
                gl.disable(glow::PROGRAM_POINT_SIZE);
            }
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, self.framebuffer);
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, target);
            if scissor {
                gl.enable(glow::SCISSOR_TEST);
            }
            gl.blit_framebuffer(
                0,
                0,
                width,
                height,
                viewport.left_px,
                viewport.from_bottom_px,
                viewport.left_px + width,
                viewport.from_bottom_px + height,
                glow::COLOR_BUFFER_BIT,
                glow::NEAREST,
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, target);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
        }
        Ok(())
    }

    fn ensure_initialized(&mut self, gl: &glow::Context) -> Result<(), String> {
        unsafe {
            if self.program.is_some() {
                return Ok(());
            }
            let version = if gl.get_parameter_string(glow::VERSION).contains("OpenGL ES") {
                "#version 300 es\nprecision highp float;\n"
            } else {
                "#version 330 core\n"
            };
            let program = gl.create_program().map_err(|e| e.to_string())?;
            let build = (|| {
                let vertex = compile_shader(
                    gl,
                    glow::VERTEX_SHADER,
                    &format!("{version}{VERTEX_SHADER}"),
                )?;
                let fragment = match compile_shader(
                    gl,
                    glow::FRAGMENT_SHADER,
                    &format!("{version}{FRAGMENT_SHADER}"),
                ) {
                    Ok(shader) => shader,
                    Err(error) => {
                        gl.delete_shader(vertex);
                        return Err(error);
                    }
                };
                gl.attach_shader(program, vertex);
                gl.attach_shader(program, fragment);
                gl.link_program(program);
                gl.delete_shader(vertex);
                gl.delete_shader(fragment);
                if !gl.get_program_link_status(program) {
                    return Err(gl.get_program_info_log(program));
                }
                Ok(())
            })();
            if let Err(error) = build {
                gl.delete_program(program);
                return Err(error);
            }
            self.program = Some(program);
            self.vao = Some(gl.create_vertex_array().map_err(|e| e.to_string())?);
            self.vbo = Some(gl.create_buffer().map_err(|e| e.to_string())?);
            gl.bind_vertex_array(self.vao);
            gl.bind_buffer(glow::ARRAY_BUFFER, self.vbo);
            let stride = mem::size_of::<Vertex>() as i32;
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, stride, 0);
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(1, 4, glow::UNSIGNED_BYTE, true, stride, 12);
            gl.bind_vertex_array(None);
            self.framebuffer = Some(gl.create_framebuffer().map_err(|e| e.to_string())?);
            self.color = Some(gl.create_texture().map_err(|e| e.to_string())?);
            self.depth = Some(gl.create_renderbuffer().map_err(|e| e.to_string())?);
            Ok(())
        }
    }

    fn ensure_size(&mut self, gl: &glow::Context, width: i32, height: i32) -> Result<(), String> {
        unsafe {
            if self.size == (width, height) {
                return Ok(());
            }
            gl.bind_texture(glow::TEXTURE_2D, self.color);
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA8 as i32,
                width,
                height,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(None),
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR as i32,
            );
            gl.bind_renderbuffer(glow::RENDERBUFFER, self.depth);
            gl.renderbuffer_storage(glow::RENDERBUFFER, glow::DEPTH_COMPONENT24, width, height);
            gl.bind_framebuffer(glow::FRAMEBUFFER, self.framebuffer);
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                self.color,
                0,
            );
            gl.framebuffer_renderbuffer(
                glow::FRAMEBUFFER,
                glow::DEPTH_ATTACHMENT,
                glow::RENDERBUFFER,
                self.depth,
            );
            if gl.check_framebuffer_status(glow::FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE {
                return Err("Could not create 3D plot framebuffer".into());
            }
            self.size = (width, height);
            Ok(())
        }
    }

    pub fn destroy(&mut self, gl: &glow::Context) {
        unsafe {
            if let Some(buffer) = self.vbo.take() {
                gl.delete_buffer(buffer);
            }
            if let Some(vao) = self.vao.take() {
                gl.delete_vertex_array(vao);
            }
            if let Some(program) = self.program.take() {
                gl.delete_program(program);
            }
            if let Some(framebuffer) = self.framebuffer.take() {
                gl.delete_framebuffer(framebuffer);
            }
            if let Some(texture) = self.color.take() {
                gl.delete_texture(texture);
            }
            if let Some(renderbuffer) = self.depth.take() {
                gl.delete_renderbuffer(renderbuffer);
            }
        }
        self.uploaded = None;
        self.size = (0, 0);
    }
}

fn compile_shader(gl: &glow::Context, kind: u32, source: &str) -> Result<glow::Shader, String> {
    unsafe {
        let shader = gl.create_shader(kind).map_err(|e| e.to_string())?;
        gl.shader_source(shader, source);
        gl.compile_shader(shader);
        if gl.get_shader_compile_status(shader) {
            Ok(shader)
        } else {
            let error = gl.get_shader_info_log(shader);
            gl.delete_shader(shader);
            Err(error)
        }
    }
}

const VERTEX_SHADER: &str = r#"
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec4 a_color;
uniform vec2 u_angles;
uniform float u_zoom;
uniform float u_aspect;
uniform float u_dpi;
uniform vec2 u_pan;
uniform int u_clustered;
uniform int u_guide;
uniform float u_depth_scale;
uniform vec4 u_plain;
uniform vec4 u_background;
out vec4 v_color;
void main() {
    float sy = sin(u_angles.x), cy = cos(u_angles.x);
    float sp = sin(u_angles.y), cp = cos(u_angles.y);
    float source_z = a_position.z * (u_guide != 0 ? 1.0 : u_depth_scale);
    float x = a_position.x * cy - source_z * sy;
    float z = a_position.x * sy + source_z * cy;
    float y = a_position.y * cp - z * sp;
    z = a_position.y * sp + z * cp;
    float depth = 3.5 - z;
    float scale = 2.4 * u_zoom / depth;
    gl_Position = vec4(x * scale / u_aspect + u_pan.x,
                       y * scale + u_pan.y, -z * 0.4, 1.0);
    gl_PointSize = clamp(8.0 * u_dpi * 3.5 / depth, 4.0 * u_dpi, 18.0 * u_dpi);
    v_color = u_guide != 0 ? u_plain : (u_clustered != 0 ? a_color : u_plain);
    if (u_guide == 0) {
        float nearness = clamp((z + 1.7) / 3.4, 0.0, 1.0);
        v_color.rgb = mix(u_background.rgb, v_color.rgb, 0.52 + 0.48 * nearness);
    }
}
"#;

const FRAGMENT_SHADER: &str = r#"
in vec4 v_color;
uniform int u_guide;
out vec4 out_color;
void main() {
    if (u_guide == 0) {
        vec2 offset = gl_PointCoord * 2.0 - 1.0;
        float radius2 = dot(offset, offset);
        if (radius2 > 1.0) discard;
    }
    out_color = v_color;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_expansion_preserves_cluster_distances_and_guide_ranges() {
        let cloud = Cloud3D::new(vec![[-1.0, 0.0, -0.1], [1.0, 0.0, 0.1]]);
        assert_eq!(cloud.positions().len(), 2);
        assert_eq!(cloud.floor_count, 20);
        assert_eq!(cloud.frame_count, 24);
        assert_eq!(cloud.depth_scale, 4.0);
        assert_eq!(cloud.vertices.len(), 2 + 20 + 24);
        assert_eq!(cloud.positions()[0][2], -0.1);
    }

    #[test]
    fn default_camera_fits_frame_in_a_narrow_plot() {
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(200.0, 300.0));
        let projection = Projection::new(Camera::default().fitted(rect), rect, 1.0);
        for x in [-GUIDE_EXTENT, GUIDE_EXTENT] {
            for y in [-GUIDE_EXTENT, GUIDE_EXTENT] {
                for z in [-GUIDE_EXTENT, GUIDE_EXTENT] {
                    let (screen, _) = projection.screen_position([x, y, z]).unwrap();
                    assert!(
                        rect.contains(screen),
                        "frame corner {x}, {y}, {z} is outside plot"
                    );
                }
            }
        }
    }

    #[test]
    fn dragging_right_moves_pc1_axis_right() {
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(600.0, 400.0));
        let mut camera = Camera::default();
        let before = Projection::new(camera, rect, 1.0)
            .screen_position([1.0, 0.0, 0.0])
            .unwrap()
            .0
            .x;
        camera.rotate(egui::vec2(20.0, 0.0));
        let after = Projection::new(camera, rect, 1.0)
            .screen_position([1.0, 0.0, 0.0])
            .unwrap()
            .0
            .x;
        assert!(after > before);
    }
}
