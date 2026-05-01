use std::time::Duration;

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph};
use ratatui::Frame;

use super::api::{
    abstractions::AI,
    bq::{BQModel, GlobalBQ},
    eps::Ep,
    rest::{get_ipv4_address, run_api},
};
use super::localization::{translate, Key, Lang};

// ── palette ──────────────────────────────────────────────────────────
const BG_DARK: Color = Color::Rgb(13, 13, 18);
const BG_SIDEBAR: Color = Color::Rgb(18, 18, 24);
const BG_SURFACE: Color = Color::Rgb(26, 26, 34);
const BG_ACTIVE: Color = Color::Rgb(35, 60, 45);
const BG_STATUS: Color = Color::Rgb(20, 20, 28);
const BG_POPUP: Color = Color::Rgb(22, 22, 30);

const FG_DIM: Color = Color::Rgb(80, 95, 85);
const FG_MUTED: Color = Color::Rgb(120, 140, 128);
const FG_BRIGHT: Color = Color::Rgb(230, 245, 235);

const ACCENT: Color = Color::Rgb(51, 218, 114);
const ACCENT_DIM: Color = Color::Rgb(30, 120, 65);
const BORDER: Color = Color::Rgb(38, 50, 42);

fn s(fg: Color, bg: Color) -> Style { Style::default().fg(fg).bg(bg) }
fn bold(fg: Color) -> Style { Style::default().fg(fg).add_modifier(Modifier::BOLD) }
fn centered(span: Span) -> Paragraph { Paragraph::new(span).alignment(Alignment::Center) }
fn at(area: Rect, y: u16) -> Rect { Rect { y, height: 1, ..area } }

// ── types ────────────────────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq)]
enum Row { Ai, ClsAi, Ep, Deploy }

struct App {
    lang: Lang,
    row: usize,
    side_btn: bool, // true = focus is on the +/- button, not the combo
    ais: Vec<AI>,
    ai_options: Vec<String>,  ai_selected: Option<usize>,  ai_open: bool,  ai_cursor: usize,
    cls_ais: Vec<AI>,
    cls_active: bool, cls_selected: Option<usize>, cls_open: bool, cls_cursor: usize,
    ep_selected: Option<Ep>,  ep_open: bool,  ep_cursor: usize,
    api_deployed: bool,
    host_url: Option<String>,
    status_msg: Option<String>,
}

impl App {
    fn new(lang: Lang) -> Self {
        let ais = BQModel::get_bqs();
        let ai_options: Vec<String> = ais.iter().map(|ai| ai.name.clone()).collect();
        let cls_ais: Vec<AI> = ais.iter().filter(|ai| ai.task == "classify").cloned().collect();
        Self {
            lang,
            row: 0, side_btn: false,
            ais, ai_options,
            ai_selected: None, ai_open: false, ai_cursor: 0,
            cls_ais,
            cls_active: false, cls_selected: None, cls_open: false, cls_cursor: 0,
            ep_selected: None, ep_open: false, ep_cursor: 0,
            api_deployed: false,
            host_url: None,
            status_msg: None,
        }
    }
    fn t(&self, key: Key) -> &'static str {
        translate(key, &self.lang)
    }
    fn rows(&self) -> Vec<Row> {
        let mut v = vec![Row::Ai];
        if self.cls_active { v.push(Row::ClsAi); }
        v.push(Row::Ep);
        if self.ai_selected.is_some() && self.ep_selected.is_some() || self.api_deployed {
            v.push(Row::Deploy);
        }
        v
    }
    fn cur_row(&self) -> Row {
        let rows = self.rows();
        rows[self.row.min(rows.len() - 1)]
    }
    fn can_add_cls(&self) -> bool {
        self.ai_selected.is_some()
            && !self.cls_active
            && !self.cls_ais.is_empty()
            && self.ai_selected.map_or(false, |i| self.ais[i].task != "classify")
    }
    fn has_side_btn(&self) -> bool {
        match self.cur_row() {
            Row::Ai => self.can_add_cls(),
            Row::ClsAi => true,
            _ => false,
        }
    }
    fn clamp(&mut self) {
        let len = self.rows().len();
        if self.row >= len { self.row = len - 1; }
        if !self.has_side_btn() { self.side_btn = false; }
    }
    fn any_open(&self) -> bool { self.ai_open || self.ep_open || self.cls_open }
}

// ── main ─────────────────────────────────────────────────────────────
pub fn run_tui(lang: Lang) -> std::io::Result<()> {
    let mut app = App::new(lang);
    ratatui::run(|terminal| loop {
        terminal.draw(|f| draw(f, &app))?;
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press && handle_input(&mut app, key.code, key.modifiers) {
                    break Ok(());
                }
            }
        }
    })
}

// ── input ────────────────────────────────────────────────────────────
fn handle_input(app: &mut App, code: KeyCode, mods: KeyModifiers) -> bool {
    if code == KeyCode::Char('c') && mods.contains(KeyModifiers::CONTROL) { return true; }
    if matches!(code, KeyCode::Char('q') | KeyCode::Esc) && !app.any_open() { return true; }

    if app.ai_open {
        let prev = app.ai_selected;
        let r = handle_dropdown(code, app.ai_options.len(), &mut app.ai_cursor, &mut app.ai_selected, &mut app.ai_open);
        if !app.ai_open && app.ai_selected != prev && app.ai_selected.is_some() {
            load_ai_model(app);
        }
        return r;
    }
    if app.cls_open {
        let cls_names: Vec<String> = app.cls_ais.iter().map(|ai| ai.name.clone()).collect();
        let prev = app.cls_selected;
        let r = handle_dropdown(code, cls_names.len(), &mut app.cls_cursor, &mut app.cls_selected, &mut app.cls_open);
        if !app.cls_open && app.cls_selected != prev && app.cls_selected.is_some() {
            load_cls_model(app);
        }
        return r;
    }
    if app.ep_open {
        let prev = app.ep_selected;
        let ep_options = [Ep::Cpu, Ep::Cuda];
        let mut temp_selected = app.ep_selected.and_then(|ep| ep_options.iter().position(|&e| e == ep));
        let r = handle_dropdown(code, ep_options.len(), &mut app.ep_cursor, &mut temp_selected, &mut app.ep_open);
        if !app.ep_open {
            app.ep_selected = temp_selected.map(|i| ep_options[i]);
        }
        if !app.ep_open && app.ep_selected != prev && app.ep_selected.is_some() {
            load_ai_model(app);
            load_cls_model(app);
        }
        return r;
    }

    let row_len = app.rows().len();
    match code {
        KeyCode::Tab | KeyCode::Down => { app.row = (app.row + 1) % row_len; app.side_btn = false; }
        KeyCode::BackTab | KeyCode::Up => { app.row = (app.row + row_len - 1) % row_len; app.side_btn = false; }
        KeyCode::Right => { if app.has_side_btn() { app.side_btn = true; } }
        KeyCode::Left  => { app.side_btn = false; }
        KeyCode::Enter => {
            if app.side_btn {
                match app.cur_row() {
                    Row::Ai => { app.cls_active = true; app.side_btn = false; app.row = 1; }
                    Row::ClsAi => { app.cls_active = false; app.cls_selected = None; BQModel::clear_second(); app.side_btn = false; app.clamp(); }
                    _ => {}
                }
            } else {
                match app.cur_row() {
                    Row::Ai => { app.ai_open = true; app.ai_cursor = app.ai_selected.unwrap_or(0); }
                    Row::ClsAi => { app.cls_open = true; app.cls_cursor = app.cls_selected.unwrap_or(0); }
                    Row::Ep => { app.ep_open = true; app.ep_cursor = app.ep_selected.map_or(0, |ep| match ep { Ep::Cuda => 1, _ => 0 }); }
                    Row::Deploy => {
                        if !app.api_deployed && app.ai_selected.is_some() && app.ep_selected.is_some() {
                            deploy_api(app);
                        }
                    }
                }
            }
        }
        _ => {}
    }
    false
}

fn handle_dropdown(code: KeyCode, len: usize, cursor: &mut usize, selected: &mut Option<usize>, open: &mut bool) -> bool {
    match code {
        KeyCode::Up   => *cursor = cursor.saturating_sub(1),
        KeyCode::Down => *cursor = (*cursor + 1).min(len.saturating_sub(1)),
        KeyCode::Enter => { *selected = Some(*cursor); *open = false; }
        KeyCode::Esc   => *open = false,
        _ => {}
    }
    false
}

fn load_ai_model(app: &mut App) {
    if let Some(ai_idx) = app.ai_selected {
        let ep = app.ep_selected.unwrap_or(Ep::Cpu);
        let model_path = app.ais[ai_idx].get_path();
        match GlobalBQ::First.set_model(&model_path, ep, None) {
            Ok(_) => { app.status_msg = Some(format!("{} {}", app.t(Key::loaded), app.ais[ai_idx].name)); }
            Err(e) => { app.status_msg = Some(format!("{}: {}", app.t(Key::error_ocurred), e)); }
        }
    }
}

fn load_cls_model(app: &mut App) {
    if let Some(cls_idx) = app.cls_selected {
        let ep = app.ep_selected.unwrap_or(Ep::Cpu);
        let model_path = app.cls_ais[cls_idx].get_path();
        match GlobalBQ::Second.set_model(&model_path, ep, None) {
            Ok(_) => { app.status_msg = Some(format!("{} {}", app.t(Key::loaded), app.cls_ais[cls_idx].name)); }
            Err(e) => { app.status_msg = Some(format!("{}: {}", app.t(Key::error_ocurred), e)); }
        }
    }
}

fn deploy_api(app: &mut App) {
    let port = 8791u16;
    tokio::spawn(async move {
        if let Err(e) = run_api(port).await {
            eprintln!("API error: {}", e);
        }
    });
    app.host_url = get_ipv4_address().map(|ip| format!("http://{}:{}", ip, port));
    app.api_deployed = true;
    app.status_msg = Some(app.t(Key::deployed_api).to_string());
}

// ── drawing ──────────────────────────────────────────────────────────
fn draw(frame: &mut Frame, app: &App) {
    frame.render_widget(Block::default().style(Style::default().bg(BG_DARK)), frame.area());
    let [body, status] = Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).areas(frame.area());
    let [sidebar, central] = Layout::horizontal([Constraint::Length(28), Constraint::Min(0)]).areas(body);

    draw_sidebar(frame, app, sidebar);
    draw_central(frame, app, central);
    draw_status_bar(frame, app, status);

    if app.ai_open { draw_dropdown_overlay(frame, app, "ai"); }
    if app.cls_open { draw_dropdown_overlay(frame, app, "cls"); }
    if app.ep_open { draw_dropdown_overlay(frame, app, "ep"); }
}

fn draw_sidebar(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default().borders(Borders::RIGHT).border_style(Style::default().fg(BORDER)).style(Style::default().bg(BG_SIDEBAR));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let cls_rows: u16 = if app.cls_active { 3 } else { 0 };
    let [head, ai, cls_area, ep, _gap, btn, _rest, hints] = Layout::vertical([
        Constraint::Length(3), Constraint::Length(3), Constraint::Length(cls_rows), Constraint::Length(3),
        Constraint::Length(2), Constraint::Length(3), Constraint::Min(0), Constraint::Length(2),
    ]).areas(inner);

    // heading
    frame.render_widget(
        Paragraph::new(Line::from(vec![Span::styled("◈ ", Style::default().fg(ACCENT)), Span::styled(app.t(Key::setup), bold(FG_BRIGHT))]))
            .alignment(Alignment::Center),
        at(head, head.y + 1),
    );

    draw_combo(frame, ai, app.t(Key::select_ai), &app.ai_options, app.ai_selected, app.cur_row() == Row::Ai && !app.side_btn);

    // [+] button to the right of the Model combo
    if app.can_add_cls() {
        let focused = app.cur_row() == Row::Ai && app.side_btn;
        draw_side_btn(frame, ai, "+", focused);
    }

    // classification AI section
    if app.cls_active {
        let cls_names: Vec<String> = app.cls_ais.iter().map(|ai| ai.name.clone()).collect();
        draw_combo(frame, cls_area, app.t(Key::select_2nd_ai), &cls_names, app.cls_selected, app.cur_row() == Row::ClsAi && !app.side_btn);
        let focused = app.cur_row() == Row::ClsAi && app.side_btn;
        draw_side_btn(frame, cls_area, "-", focused);
    }

    let ep_options: &[&str] = &[Ep::Cpu.name(), Ep::Cuda.name()];
    let ep_selected_idx = app.ep_selected.and_then(|ep| ep_options.iter().position(|&name| name == ep.name()));
    draw_combo(frame, ep, app.t(Key::select_ep), ep_options, ep_selected_idx, app.cur_row() == Row::Ep);

    // deploy button — only visible when both AI and processor are chosen
    let can_deploy = app.ai_selected.is_some() && app.ep_selected.is_some();
    if can_deploy || app.api_deployed {
        let focused = app.cur_row() == Row::Deploy;
        let label_text = if app.api_deployed { app.t(Key::api_live) } else { app.t(Key::deploy_api) };
        let label = &format!(" {}", label_text);
        let style = match (focused, app.api_deployed) {
            (true, true)   => s(FG_BRIGHT, Color::Rgb(25, 70, 40)),
            (_, true)      => s(ACCENT, Color::Rgb(20, 55, 35)),
            (true, false)  => s(FG_BRIGHT, BG_ACTIVE),
            (false, false) => s(FG_MUTED, BG_SURFACE),
        };
        frame.render_widget(
            Paragraph::new(Span::styled(label, style)).alignment(Alignment::Center),
            Rect { x: btn.x + 2, y: btn.y + 1, width: btn.width.saturating_sub(4), height: 1 },
        );
    }

    // hints
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("↑↓", Style::default().fg(ACCENT_DIM)), Span::styled(format!(" {}  ", app.t(Key::nav_hint)), Style::default().fg(FG_DIM)),
            Span::styled("⏎", Style::default().fg(ACCENT_DIM)),  Span::styled(format!(" {}", app.t(Key::select_hint)), Style::default().fg(FG_DIM)),
        ])).alignment(Alignment::Center),
        at(hints, hints.y + 1),
    );
}

fn draw_side_btn(frame: &mut Frame, row_area: Rect, symbol: &str, focused: bool) {
    let (fg, bg) = if focused { (ACCENT, BG_ACTIVE) } else { (FG_DIM, BG_SURFACE) };
    let bx = row_area.x + row_area.width.saturating_sub(3);
    frame.render_widget(
        Paragraph::new(Span::styled(format!("[{symbol}]"), s(fg, bg))),
        Rect { x: bx, y: row_area.y + 1, width: 3, height: 1 },
    );
}

fn draw_combo(frame: &mut Frame, area: Rect, label: &str, options: &[impl AsRef<str>], selected: Option<usize>, focused: bool) {
    if area.height < 2 { return; }

    let lbl = Rect { x: area.x + 2, y: area.y, width: area.width.saturating_sub(2), height: 1 };
    let combo = Rect { x: area.x + 2, y: area.y + 1, width: area.width.saturating_sub(4), height: 1 };

    let lc = if focused { ACCENT } else { FG_DIM };
    frame.render_widget(Paragraph::new(Span::styled(label, Style::default().fg(lc))), lbl);

    let text = selected.map_or("—".into(), |i| options[i].as_ref().to_owned());
    let (fg, bg, ic) = if focused { (FG_BRIGHT, BG_ACTIVE, ACCENT) } else { (FG_MUTED, BG_SURFACE, FG_DIM) };

    let w = combo.width as usize;
    let chev = " ▾";
    let max = w.saturating_sub(chev.len());
    let trunc: String = text.chars().take(max).collect();
    let pad = max.saturating_sub(trunc.len());

    frame.render_widget(Paragraph::new(Line::from(vec![
        Span::styled(format!(" {trunc}{}", " ".repeat(pad)), s(fg, bg)),
        Span::styled(chev, s(ic, bg)),
    ])), combo);
}

fn draw_central(frame: &mut Frame, app: &App, area: Rect) {
    let cy = area.y + area.height / 2;
    if app.api_deployed {
        frame.render_widget(centered(Span::styled("●", bold(ACCENT))), at(area, cy.saturating_sub(1)));
        frame.render_widget(centered(Span::styled(app.t(Key::deployed_api), Style::default().fg(FG_MUTED))), at(area, cy + 1));
        if let Some(url) = &app.host_url {
            let focused = app.cur_row() == Row::Deploy;
            if focused {
                frame.render_widget(centered(Span::styled(url.as_str(), Style::default().fg(FG_BRIGHT))), at(area, cy + 3));
            } else {
                frame.render_widget(centered(Span::styled(app.t(Key::focus_deploy_to_reveal_ip), Style::default().fg(FG_DIM))), at(area, cy + 3));
            }
        }
    } else {
        if cy >= 2 {
            frame.render_widget(centered(Span::styled("◇", bold(FG_DIM))), at(area, cy - 2));
        }
        frame.render_widget(centered(Span::styled(app.t(Key::no_api_running), Style::default().fg(FG_DIM))), at(area, cy));
        frame.render_widget(centered(Span::styled(app.t(Key::select_model_and_deploy), Style::default().fg(Color::Rgb(55, 55, 70)))), at(area, cy + 1));
    }
}

fn draw_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    let model = app.ai_selected.map_or(String::new(), |i| format!("  {}  ", app.ai_options[i]));
    let cls = app.cls_selected.map_or(String::new(), |i| format!("+ {}  ", app.cls_ais[i].name));
    let ep = app.ep_selected.map_or(String::new(), |ep| format!("  {}  ", ep.name()));
    let (api_label, api_fg) = if app.api_deployed { (" ● LIVE ", ACCENT) } else { (" ○ OFF ", FG_DIM) };

    frame.render_widget(Paragraph::new(Line::from(vec![
        Span::styled(" BoquilaHUB ", s(ACCENT, BG_STATUS).add_modifier(Modifier::BOLD)),
        Span::styled("│", s(BORDER, BG_STATUS)),
        Span::styled(api_label, s(api_fg, BG_STATUS)),
        Span::styled("│", s(BORDER, BG_STATUS)),
        Span::styled(model, s(FG_MUTED, BG_STATUS)),
        Span::styled(cls, s(FG_DIM, BG_STATUS)),
        Span::styled(ep, s(FG_DIM, BG_STATUS)),
    ])).style(Style::default().bg(BG_STATUS)), area);
}

fn draw_dropdown_overlay(frame: &mut Frame, app: &App, which: &str) {
    let cls_names: Vec<String> = app.cls_ais.iter().map(|ai| ai.name.clone()).collect();
    let ep_options: Vec<String> = [Ep::Cpu, Ep::Cuda].iter().map(|ep| ep.name().to_string()).collect();
    let ep_selected_idx = app.ep_selected.and_then(|ep| [Ep::Cpu, Ep::Cuda].iter().position(|&e| e == ep));
    let (options, cursor, selected, title) = match which {
        "ai"  => (&app.ai_options, app.ai_cursor, app.ai_selected, app.t(Key::select_ai)),
        "cls" => (&cls_names, app.cls_cursor, app.cls_selected, app.t(Key::select_2nd_ai)),
        _     => (&ep_options, app.ep_cursor, ep_selected_idx, app.t(Key::select_ep)),
    };

    let cls_offset: u16 = if app.cls_active { 3 } else { 0 };
    let y = match which {
        "ai"  => 4,
        "cls" => 7,
        _     => 7 + cls_offset,
    };
    let max_h = frame.area().height.saturating_sub(y);
    if max_h < 3 { return; }
    let popup_h = (options.len() as u16 + 2).min(max_h);
    let visible = popup_h.saturating_sub(2) as usize; // inner rows (minus border)
    let scroll = if cursor >= visible { cursor - visible + 1 } else { 0 };
    let popup = Rect { x: 2, y, width: 24, height: popup_h };
    frame.render_widget(Clear, popup);

    let items: Vec<ListItem> = options.iter().enumerate().skip(scroll).map(|(i, opt)| {
        let cur = i == cursor;
        let sel = selected == Some(i);
        let base = if cur { bold(FG_BRIGHT).bg(BG_ACTIVE) } else { s(FG_MUTED, BG_POPUP) };
        ListItem::new(Line::from(vec![
            Span::styled(if sel { "● " } else { "  " }, if sel { base.fg(ACCENT) } else { base }),
            Span::styled(opt.as_str(), base),
        ]))
    }).collect();

    frame.render_widget(List::new(items).block(
        Block::bordered()
            .title(Line::from(vec![
                Span::styled(" ", Style::default().fg(ACCENT)),
                Span::styled(title, bold(FG_BRIGHT)),
                Span::styled(" ", Style::default().fg(ACCENT)),
            ]))
            .border_style(Style::default().fg(ACCENT_DIM))
            .style(Style::default().bg(BG_POPUP)),
    ), popup);
}
