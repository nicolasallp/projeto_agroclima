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
