"""Testes de interface com ``streamlit.testing.v1.AppTest``.

Rodar:  python -m pytest -q tests/test_app.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import config
import model_utils as mu

APP = str(Path(__file__).resolve().parents[1] / "app.py")
TIMEOUT = 300


def _labels(at: AppTest) -> list[str]:
    widgets = [*at.selectbox, *at.number_input, *at.slider, *at.radio, *at.text_input, *at.toggle]
    return [w.label for w in widgets]


def _rodar_modo(modo: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.session_state["modo_ativo"] = modo
    at.session_state["_w_modo"] = modo
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


@pytest.mark.parametrize("modo", config.MODOS)
def test_modo_abre_sem_erro_e_sem_seletor_de_ano(modo):
    at = _rodar_modo(modo)
    assert not at.error, [e.value for e in at.error]
    assert all("ano" not in lbl.lower().split() for lbl in _labels(at)), _labels(at)
    assert any(config.NOTA_ANO in c.value for c in at.caption)


def test_modo1_nao_roda_quando_outro_modo_esta_ativo(monkeypatch):
    chamadas = []
    original = mu.prever_mapa_produto
    monkeypatch.setattr(mu, "prever_mapa_produto", lambda *a, **k: chamadas.append(a) or original(*a, **k))
    _rodar_modo(config.MODOS[1])
    _rodar_modo(config.MODOS[2])
    assert chamadas == []


def test_modo3_calcular_e_restaurar():
    at = _rodar_modo(config.MODOS[2])
    codigo = at.session_state["m3_codigo"]
    chave = f"m3_chuva_total_{codigo}"
    base = at.session_state[chave]
    # Calcula
    at.button(key="FormSubmitter:m3_form-Calcular").click().run()
    assert not at.exception
    assert at.session_state["m3_resultado"]["ano"] == config.ANO_REFERENCIA
    assert any("Rendimento previsto" in m.label for m in at.metric)
    # Altera, submete o form e restaura
    at.number_input(key=chave).set_value(base + 500)
    at.button(key="FormSubmitter:m3_form-Calcular").click().run()
    assert not at.exception
    assert at.session_state[chave] == pytest.approx(base + 500)
    at.button(key="m3_btn_restaurar").click().run()
    assert not at.exception
    assert at.session_state[chave] == pytest.approx(base)


def test_modo3_lista_apenas_produtos_validos():
    at = _rodar_modo(config.MODOS[2])
    refs = mu.carregar_referencias()
    for _ in range(2):
        codigo = at.session_state["m3_codigo"]
        opcoes = at.selectbox(key="_w_m3_produto").options
        assert sorted(opcoes) == refs.produtos_validos(codigo)
        outro = [c for c in at.selectbox(key="_w_m3_codigo").options if c != codigo][0]
        at.selectbox(key="_w_m3_codigo").set_value(outro).run()
        assert not at.exception


def test_modo2_painel_municipio_via_estado():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.session_state["modo_ativo"] = config.MODOS[1]
    at.session_state["_w_modo"] = config.MODOS[1]
    at.session_state["m2_codigo_ibge"] = "3550308"  # São Paulo - SP
    at.run()
    assert not at.exception
    assert any("Município: São Paulo" in m.value for m in at.markdown)


def test_modo2_troca_de_nivel_limpa_selecao_e_mostra_mensagem():
    at = _rodar_modo(config.MODOS[1])
    at.session_state["m2_codigo_ibge"] = "3550308"
    at.run()
    assert any("Município: São Paulo" in m.value for m in at.markdown)

    # Troca para Estados: limpa seleção e mostra mensagem do nível Estados
    at.segmented_control(key="m2_nivel").set_value("Estados").run()
    assert not at.exception
    assert "m2_codigo_ibge" not in at.session_state
    assert at.session_state["m2_regiao"] == "(Brasil)"
    assert at.session_state["m2_uf"] == "(Todos)"
    assert any("Clique em um estado no mapa" in i.value for i in at.info)

    # Troca para Regiões: mostra mensagem do nível Regiões
    at.segmented_control(key="m2_nivel").set_value("Regiões").run()
    assert not at.exception
    assert any("Clique em uma região no mapa" in i.value for i in at.info)


def test_modo2_clique_estado_e_regiao_igual_ao_dropdown_sem_reenquadrar():
    at = _rodar_modo(config.MODOS[1])
    at.segmented_control(key="m2_nivel").set_value("Estados").run()
    assert not at.exception

    for uf_alvo, reg_esperada in (("PA", "Norte"), ("DF", "Centro-Oeste")):
        at.session_state["m2_enquadramento"] = (None, None)
        at.session_state["m2_regiao"] = reg_esperada
        at.session_state["m2_uf"] = uf_alvo
        at.session_state["m2_painel"] = "area"
        at.run()
        assert not at.exception
        assert at.session_state["m2_enquadramento"] == (None, None)
        assert at.selectbox(key="_w_m2_regiao").value == reg_esperada
        assert at.selectbox(key="_w_m2_uf").value == uf_alvo
        df_clique = at.dataframe[0].value.copy()

        # Limpa para "(Brasil)" e seleciona a mesma área pelos dropdowns (que reenquadram o mapa)
        at.selectbox(key="_w_m2_regiao").set_value("(Brasil)").run()
        at.selectbox(key="_w_m2_regiao").set_value(reg_esperada).run()
        at.selectbox(key="_w_m2_uf").set_value(uf_alvo).run()
        assert at.session_state["m2_enquadramento"] == (reg_esperada, uf_alvo)
        df_dropdown = at.dataframe[0].value.copy()
        assert df_clique.equals(df_dropdown)
        at.selectbox(key="_w_m2_regiao").set_value("(Brasil)").run()


def test_callback_ao_clicar_mapa_estados_e_regioes(monkeypatch):
    import streamlit as st

    ns: dict = {}
    exec(compile(Path(APP).read_text(encoding="utf-8").replace("\nmain()\n", "\n"), APP, "exec"), ns)

    fake_state = {
        "m2_nivel_ativo": "Estados",
        "m2_enquadramento": (None, None),
        "m2_mapa_estados": {"selection": {"points": [{"customdata": ["PA", "Pará (PA)"]}]}},
    }
    monkeypatch.setattr(st, "session_state", fake_state)
    ns["_ao_clicar_mapa"]()
    assert fake_state["m2_regiao"] == "Norte"
    assert fake_state["m2_uf"] == "PA"
    assert fake_state["m2_painel"] == "area"
    assert fake_state["m2_enquadramento"] == (None, None)

    fake_state["m2_mapa_estados"] = {"selection": {"points": [{"customdata": ["DF", "Distrito Federal (DF)"]}]}}
    ns["_ao_clicar_mapa"]()
    assert fake_state["m2_regiao"] == "Centro-Oeste"
    assert fake_state["m2_uf"] == "DF"
    assert fake_state["m2_enquadramento"] == (None, None)

    fake_state["m2_nivel_ativo"] = "Regiões"
    fake_state["m2_mapa_regioes"] = {"selection": {"points": [{"customdata": ["Sul", "Região Sul"]}]}}
    ns["_ao_clicar_mapa"]()
    assert fake_state["m2_regiao"] == "Sul"
    assert fake_state["m2_uf"] == "(Todos)"
    assert fake_state["m2_painel"] == "area"
    assert fake_state["m2_enquadramento"] == (None, None)
