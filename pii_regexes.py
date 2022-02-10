"""
Copyright, 2021-2022 Ontocord, LLC, All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import stdnum
import re, regex
from stdnum import (bic, bitcoin, casrn, cusip, ean, figi, grid, gs1_128, iban, \
                    imei, imo, imsi, isan, isbn, isil, isin, ismn, iso11649, iso6346, \
                    iso9362, isrc, issn, lei,  mac, meid, vatin)
from stdnum.ad import nrt
from stdnum.al import nipt
from stdnum.ar import dni
from stdnum.ar import cbu
from stdnum.ar import cuit
from stdnum.at import businessid
from stdnum.at import tin
from stdnum.at import vnr
from stdnum.at import postleitzahl
from stdnum.at import uid
from stdnum.au import abn
from stdnum.au import acn
from stdnum.au import tfn
from stdnum.be import iban
#from stdnum.be import nn
from stdnum.be import vat
from stdnum.bg import vat
from stdnum.bg import egn
from stdnum.bg import pnf
from stdnum.br import cnpj
from stdnum.br import cpf
from stdnum.by import unp
from stdnum.ca import sin
from stdnum.ca import bn
from stdnum.ch import ssn
from stdnum.ch import vat
from stdnum.ch import uid
from stdnum.ch import esr
from stdnum.cl import rut
from stdnum.cn import ric
from stdnum.cn import uscc
from stdnum.co import nit
from stdnum.cr import cpj
from stdnum.cr import cr
from stdnum.cr import cpf
from stdnum.cu import ni
from stdnum.cy import vat
from stdnum.cz import dic
from stdnum.cz import rc
from stdnum.de import vat
from stdnum.de import handelsregisternummer
from stdnum.de import wkn
from stdnum.de import stnr
from stdnum.de import idnr
from stdnum.dk import cvr
from stdnum.dk import cpr
from stdnum.do import rnc
from stdnum.do import ncf
from stdnum.do import cedula
from stdnum.ec import ruc
from stdnum.ec import ci
from stdnum.ee import ik
from stdnum.ee import registrikood
from stdnum.ee import kmkr
from stdnum.es import iban
from stdnum.es import ccc
from stdnum.es import cif
from stdnum.es import dni
from stdnum.es import cups
from stdnum.es import referenciacatastral
from stdnum.es import nie
from stdnum.es import nif
from stdnum.eu import banknote
from stdnum.eu import eic
from stdnum.eu import vat
from stdnum.eu import at_02
from stdnum.eu import nace
from stdnum.fi import associationid
from stdnum.fi import veronumero
from stdnum.fi import hetu
from stdnum.fi import ytunnus
from stdnum.fi import alv
from stdnum.fr import siret
from stdnum.fr import tva
from stdnum.fr import nir
from stdnum.fr import nif
from stdnum.fr import siren
from stdnum.gb import upn
from stdnum.gb import vat
from stdnum.gb import nhs
from stdnum.gb import utr
from stdnum.gb import sedol
from stdnum.gr import vat
from stdnum.gr import amka
from stdnum.gt import nit
from stdnum.hr import oib
from stdnum.hu import anum
from stdnum.id import npwp
from stdnum.ie import vat
from stdnum.ie import pps
from stdnum.il import hp
from stdnum.il import idnr
from stdnum.in_ import epic
from stdnum.in_ import gstin
from stdnum.in_ import pan
from stdnum.in_ import aadhaar
from stdnum.is_ import kennitala
from stdnum.is_ import vsk
from stdnum.it import codicefiscale
from stdnum.it import aic
from stdnum.it import iva
from stdnum.jp import cn
from stdnum.kr import rrn
from stdnum.kr import brn
from stdnum.li import peid
from stdnum.lt import pvm
from stdnum.lt import asmens
from stdnum.lu import tva
from stdnum.lv import pvn
from stdnum.mc import tva
from stdnum.md import idno
from stdnum.me import iban
from stdnum.mt import vat
from stdnum.mu import nid
from stdnum.mx import curp
from stdnum.mx import rfc
from stdnum.my import nric
from stdnum.nl import bsn
from stdnum.nl import brin
from stdnum.nl import onderwijsnummer
from stdnum.nl import btw
from stdnum.nl import postcode
from stdnum.no import mva
from stdnum.no import iban
from stdnum.no import kontonr
from stdnum.no import fodselsnummer
from stdnum.no import orgnr
from stdnum.nz import bankaccount
from stdnum.nz import ird
from stdnum.pe import cui
from stdnum.pe import ruc
from stdnum.pl import pesel
from stdnum.pl import nip
from stdnum.pl import regon
from stdnum.pt import cc
from stdnum.pt import nif
from stdnum.py import ruc
from stdnum.ro import onrc
from stdnum.ro import cui
from stdnum.ro import cf
from stdnum.ro import cnp
from stdnum.rs import pib
from stdnum.ru import inn
from stdnum.se import vat
from stdnum.se import personnummer
from stdnum.se import postnummer
from stdnum.se import orgnr
from stdnum.sg import uen
from stdnum.si import ddv
from stdnum.sk import rc
from stdnum.sk import dph
from stdnum.sm import coe
from stdnum.sv import nit
from stdnum.th import pin
from stdnum.th import tin
from stdnum.th import moa
from stdnum.tr import vkn
from stdnum.tr import tckimlik
from stdnum.tw import ubn
from stdnum.ua import rntrc
from stdnum.ua import edrpou
from stdnum.us import ssn
from stdnum.us import atin
from stdnum.us import rtn
from stdnum.us import tin
from stdnum.us import ein
from stdnum.us import itin
from stdnum.us import ptin
from stdnum.uy import rut
from stdnum.ve import rif
from stdnum.vn import mst
from stdnum.za import tin
from stdnum.za import idnr

country_to_lang = {
  #TODO
}

stdnum_mapper = {
    'ad.nrt':  stdnum.ad.nrt.validate,
    'al.nipt':  stdnum.al.nipt.validate,
    'ar.dni':  stdnum.ar.dni.validate,
    'ar.cbu':  stdnum.ar.cbu.validate,
    'ar.cuit':  stdnum.ar.cuit.validate,
    'at.businessid':  stdnum.at.businessid.validate,
    'at.tin':  stdnum.at.tin.validate,
    'at.vnr':  stdnum.at.vnr.validate,
    'at.postleitzahl':  stdnum.at.postleitzahl.validate,
    'at.uid':  stdnum.at.uid.validate,
    'au.abn':  stdnum.au.abn.validate,
    'au.acn':  stdnum.au.acn.validate,
    'au.tfn':  stdnum.au.tfn.validate,
    'be.iban':  stdnum.be.iban.validate,
    #'be.nn':  stdnum.be.nn.validate,
    'be.vat':  stdnum.be.vat.validate,
    'bg.vat':  stdnum.bg.vat.validate,
    'bg.egn':  stdnum.bg.egn.validate,
    'bg.pnf':  stdnum.bg.pnf.validate,
    'br.cnpj':  stdnum.br.cnpj.validate,
    'br.cpf':  stdnum.br.cpf.validate,
    'by.unp':  stdnum.by.unp.validate,
    'ca.sin':  stdnum.ca.sin.validate,
    'ca.bn':  stdnum.ca.bn.validate,
    'ch.ssn':  stdnum.ch.ssn.validate,
    'ch.vat':  stdnum.ch.vat.validate,
    'ch.uid':  stdnum.ch.uid.validate,
    'ch.esr':  stdnum.ch.esr.validate,
    'cl.rut':  stdnum.cl.rut.validate,
    'cn.ric':  stdnum.cn.ric.validate,
    'cn.uscc':  stdnum.cn.uscc.validate,
    'co.nit':  stdnum.co.nit.validate,
    'cr.cpj':  stdnum.cr.cpj.validate,
    'cr.cr':  stdnum.cr.cr.validate,
    'cr.cpf':  stdnum.cr.cpf.validate,
    'cu.ni':  stdnum.cu.ni.validate,
    'cy.vat':  stdnum.cy.vat.validate,
    'cz.dic':  stdnum.cz.dic.validate,
    'cz.rc':  stdnum.cz.rc.validate,
    'de.vat':  stdnum.de.vat.validate,
    'de.handelsregisternummer':  stdnum.de.handelsregisternummer.validate,
    'de.wkn':  stdnum.de.wkn.validate,
    'de.stnr':  stdnum.de.stnr.validate,
    'de.idnr':  stdnum.de.idnr.validate,
    'dk.cvr':  stdnum.dk.cvr.validate,
    'dk.cpr':  stdnum.dk.cpr.validate,
    'do.rnc':  stdnum.do.rnc.validate,
    'do.ncf':  stdnum.do.ncf.validate,
    'do.cedula':  stdnum.do.cedula.validate,
    'ec.ruc':  stdnum.ec.ruc.validate,
    'ec.ci':  stdnum.ec.ci.validate,
    'ee.ik':  stdnum.ee.ik.validate,
    'ee.registrikood':  stdnum.ee.registrikood.validate,
    'ee.kmkr':  stdnum.ee.kmkr.validate,
    'es.iban':  stdnum.es.iban.validate,
    'es.ccc':  stdnum.es.ccc.validate,
    'es.cif':  stdnum.es.cif.validate,
    'es.dni':  stdnum.es.dni.validate,
    'es.cups':  stdnum.es.cups.validate,
    'es.referenciacatastral':  stdnum.es.referenciacatastral.validate,
    'es.nie':  stdnum.es.nie.validate,
    'es.nif':  stdnum.es.nif.validate,
    'eu.banknote':  stdnum.eu.banknote.validate,
    'eu.eic':  stdnum.eu.eic.validate,
    'eu.vat':  stdnum.eu.vat.validate,
    'eu.at_02':  stdnum.eu.at_02.validate,
    'eu.nace':  stdnum.eu.nace.validate,
    'fi.associationid':  stdnum.fi.associationid.validate,
    'fi.veronumero':  stdnum.fi.veronumero.validate,
    'fi.hetu':  stdnum.fi.hetu.validate,
    'fi.ytunnus':  stdnum.fi.ytunnus.validate,
    'fi.alv':  stdnum.fi.alv.validate,
    'fr.siret':  stdnum.fr.siret.validate,
    'fr.tva':  stdnum.fr.tva.validate,
    'fr.nir':  stdnum.fr.nir.validate,
    'fr.nif':  stdnum.fr.nif.validate,
    'fr.siren':  stdnum.fr.siren.validate,
    'gb.upn':  stdnum.gb.upn.validate,
    'gb.vat':  stdnum.gb.vat.validate,
    'gb.nhs':  stdnum.gb.nhs.validate,
    'gb.utr':  stdnum.gb.utr.validate,
    'gb.sedol':  stdnum.gb.sedol.validate,
    'gr.vat':  stdnum.gr.vat.validate,
    'gr.amka':  stdnum.gr.amka.validate,
    'gt.nit':  stdnum.gt.nit.validate,
    'hr.oib':  stdnum.hr.oib.validate,
    'hu.anum':  stdnum.hu.anum.validate,
    'id.npwp':  stdnum.id.npwp.validate,
    'ie.vat':  stdnum.ie.vat.validate,
    'ie.pps':  stdnum.ie.pps.validate,
    'il.hp':  stdnum.il.hp.validate,
    'il.idnr':  stdnum.il.idnr.validate,
    'in_.epic':  stdnum.in_.epic.validate,
    'in_.gstin':  stdnum.in_.gstin.validate,
    'in_.pan':  stdnum.in_.pan.validate,
    'in_.aadhaar':  stdnum.in_.aadhaar.validate,
    'is_.kennitala':  stdnum.is_.kennitala.validate,
    'is_.vsk':  stdnum.is_.vsk.validate,
    'it.codicefiscale':  stdnum.it.codicefiscale.validate,
    'it.aic':  stdnum.it.aic.validate,
    'it.iva':  stdnum.it.iva.validate,
    'jp.cn':  stdnum.jp.cn.validate,
    'kr.rrn':  stdnum.kr.rrn.validate,
    'kr.brn':  stdnum.kr.brn.validate,
    'li.peid':  stdnum.li.peid.validate,
    'lt.pvm':  stdnum.lt.pvm.validate,
    'lt.asmens':  stdnum.lt.asmens.validate,
    'lu.tva':  stdnum.lu.tva.validate,
    'lv.pvn':  stdnum.lv.pvn.validate,
    'mc.tva':  stdnum.mc.tva.validate,
    'md.idno':  stdnum.md.idno.validate,
    'me.iban':  stdnum.me.iban.validate,
    'mt.vat':  stdnum.mt.vat.validate,
    'mu.nid':  stdnum.mu.nid.validate,
    'mx.curp':  stdnum.mx.curp.validate,
    'mx.rfc':  stdnum.mx.rfc.validate,
    'my.nric':  stdnum.my.nric.validate,
    'nl.bsn':  stdnum.nl.bsn.validate,
    'nl.brin':  stdnum.nl.brin.validate,
    'nl.onderwijsnummer':  stdnum.nl.onderwijsnummer.validate,
    'nl.btw':  stdnum.nl.btw.validate,
    'nl.postcode':  stdnum.nl.postcode.validate,
    'no.mva':  stdnum.no.mva.validate,
    'no.iban':  stdnum.no.iban.validate,
    'no.kontonr':  stdnum.no.kontonr.validate,
    'no.fodselsnummer':  stdnum.no.fodselsnummer.validate,
    'no.orgnr':  stdnum.no.orgnr.validate,
    'nz.bankaccount':  stdnum.nz.bankaccount.validate,
    'nz.ird':  stdnum.nz.ird.validate,
    'pe.cui':  stdnum.pe.cui.validate,
    'pe.ruc':  stdnum.pe.ruc.validate,
    'pl.pesel':  stdnum.pl.pesel.validate,
    'pl.nip':  stdnum.pl.nip.validate,
    'pl.regon':  stdnum.pl.regon.validate,
    'pt.cc':  stdnum.pt.cc.validate,
    'pt.nif':  stdnum.pt.nif.validate,
    'py.ruc':  stdnum.py.ruc.validate,
    'ro.onrc':  stdnum.ro.onrc.validate,
    'ro.cui':  stdnum.ro.cui.validate,
    'ro.cf':  stdnum.ro.cf.validate,
    'ro.cnp':  stdnum.ro.cnp.validate,
    'rs.pib':  stdnum.rs.pib.validate,
    'ru.inn':  stdnum.ru.inn.validate,
    'se.vat':  stdnum.se.vat.validate,
    'se.personnummer':  stdnum.se.personnummer.validate,
    'se.postnummer':  stdnum.se.postnummer.validate,
    'se.orgnr':  stdnum.se.orgnr.validate,
    'sg.uen':  stdnum.sg.uen.validate,
    'si.ddv':  stdnum.si.ddv.validate,
    'sk.rc':  stdnum.sk.rc.validate,
    'sk.dph':  stdnum.sk.dph.validate,
    'sm.coe':  stdnum.sm.coe.validate,
    'sv.nit':  stdnum.sv.nit.validate,
    'th.pin':  stdnum.th.pin.validate,
    'th.tin':  stdnum.th.tin.validate,
    'th.moa':  stdnum.th.moa.validate,
    'tr.vkn':  stdnum.tr.vkn.validate,
    'tr.tckimlik':  stdnum.tr.tckimlik.validate,
    'tw.ubn':  stdnum.tw.ubn.validate,
    'ua.rntrc':  stdnum.ua.rntrc.validate,
    'ua.edrpou':  stdnum.ua.edrpou.validate,
    'us.ssn':  stdnum.us.ssn.validate,
    'us.atin':  stdnum.us.atin.validate,
    'us.rtn':  stdnum.us.rtn.validate,
    'us.tin':  stdnum.us.tin.validate,
    'us.ein':  stdnum.us.ein.validate,
    'us.itin':  stdnum.us.itin.validate,
    'us.ptin':  stdnum.us.ptin.validate,
    'uy.rut':  stdnum.uy.rut.validate,
    've.rif':  stdnum.ve.rif.validate,
    'vn.mst':  stdnum.vn.mst.validate,
    'za.tin':  stdnum.za.tin.validate,
    'za.idnr':  stdnum.za.idnr.validate,
    'bic':  stdnum.bic.validate,
    'bitcoin':  stdnum.bitcoin.validate,
    'casrn':  stdnum.casrn.validate,
    'cusip':  stdnum.cusip.validate,
    'ean':  stdnum.ean.validate,
    'figi':  stdnum.figi.validate,
    'grid':  stdnum.grid.validate,
    'gs1_128':  stdnum.gs1_128.validate,
    'iban':  stdnum.iban.validate,
    'imei':  stdnum.imei.validate,
    'imo':  stdnum.imo.validate,
    'imsi':  stdnum.imsi.validate,
    'isan':  stdnum.isan.validate,
    'isbn':  stdnum.isbn.validate,
    'isil':  stdnum.isil.validate,
    'isin':  stdnum.isin.validate,
    'ismn':  stdnum.ismn.validate,
    'iso11649':  stdnum.iso11649.validate,
    'iso6346':  stdnum.iso6346.validate,
    'iso9362':  stdnum.iso9362.validate,
    'isrc':  stdnum.isrc.validate,
    'issn':  stdnum.issn.validate,
    'lei':  stdnum.lei.validate,
    'mac':  stdnum.mac.validate,
    'meid':  stdnum.meid.validate,
    'vatin':  stdnum.vatin.validate,
}

def id_2_stdnum_type(text):
  #not PII = isil, isbn, isan, imo, gs1_128, grid, figi, ean, casrn, 
  #cusip number probaly PII?
  stdnum_type = []
  for id_type, validate in stdnum_mapper.items():
    try:
      found = validate(text)
    except:
      found = False
    if found:
      stdnum_type.append (id_type)
  return stdnum_type

#from https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py which is under the MIT License
# see also for ICD https://stackoverflow.com/questions/5590862/icd9-regex-pattern - but this could be totally wrong!
# we do regex in this order in order to not capture ner inside domain names and email addresses.
#NORP, AGE and DISEASE regexes are just test cases. We will use transformers and rules to detect these.
regex_rulebase = {
    "NORP": {
      "en": [(re.compile(r"upper class|middle class|working class|lower class", re.IGNORECASE), None),],
    },
    "AGE": {
      "en": [
          (
              re.compile(
                  r"\S+ years old|\S+\-years\-old|\S+ year old|\S+\-year\-old|born [ ][\d][\d]+[\\ /.][\d][\d][\\ /.][\d][\d]+|died [ ][\d][\d]+[\\ /.][\d][\d][\\ /.][\d][\d]+", re.IGNORECASE
              ),
              None,
          )
      ],
      "zh": [(regex.compile(r"\d{1,3}歲|岁"), None)],
    },
    # Some of this code from https://github.com/bigscience-workshop/data_tooling/blob/master/ac_dc/anonymization.py which is under the Apache 2 license
    "ADDRESS": {
      "en": [
              #https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py
              (
                  re.compile(
                      r"\d{1,4} [\w\s]{1,20} (?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|park|parkway|pkwy|circle|cir|boulevard|blvd)\W?(?=\s|$).*\b\d{5}(?:[-\s]\d{4})?\b|\d{1,4} [\w\s]{1,20} (?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|park|parkway|pkwy|circle|cir|boulevard|blvd)\W?(?=\s|$)", re.IGNORECASE
                  ),
                  None,
              ),
              (
                  re.compile(r"P\.? ?O\.? Box \d+"), None
              )
      ],

      "zh": [
          (
              regex.compile(
                  r"((\p{Han}{1,3}(自治区|省))?\p{Han}{1,4}((?<!集)市|县|州)\p{Han}{1,10}[路|街|道|巷](\d{1,3}[弄|街|巷])?\d{1,4}号)"
              ),
              None,
          ),
          (
              regex.compile(
                  r"(?<zipcode>(^\d{5}|^\d{3})?)(?<city>\D+[縣市])(?<district>\D+?(市區|鎮區|鎮市|[鄉鎮市區]))(?<others>.+)"
              ),
              None,
          ),
      ],
    },
    "DISEASE": {
      "en": [(re.compile("diabetes|cancer|HIV|AIDS|Alzheimer's|Alzheimer|heart disease", re.IGNORECASE), None)],
    },
    # many of the id_regex are from Presidio which is licensed under the MIT License
    # https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/aba_routing_recognizer.py
    # https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/au_abn_recognizer.py
    # https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/us_passport_recognizer.py
    # https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/medical_license_recognizer.py
    # https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/es_nif_recognizer.py
    "ID": {
      "en": [
              (
                re.compile(r"\b[0123678]\d{3}-\d{4}-\d\b"),
                (
                    "aba",
                    "routing",
                    "abarouting",
                    "association",
                    "bankrouting",
                ),
              ),
              (
                  re.compile(r"(\b[0-9]{9}\b)"),
                  (
                      "us",
                      "united",
                      "states",
                      "passport",
                      "passport#",
                      "travel",
                      "document",
                  ),
              ),
              (
                  re.compile(r"[a-zA-Z]{2}\d{7}|[a-zA-Z]{1}9\d{7}"),
                  ("medical", "certificate", "DEA"),
              ),
              (re.compile(r"\d{3}\s\d{3}\s\d{3}"), None),
              (
                  re.compile(
                      r"GB\s?\d{6}\s?\w|GB\d{3}\s\d{3}\s\d{2}\s\d{3}|GBGD\d{3}|GBHA\d{3}}|GB\d{3} \d{4} \d{2}(?: \d{3})?|GB(?:GD|HA)\d{3}"
                  ),
                  None,
              ),
              (re.compile(r"IE\d[1-9]\d{5}\d[1-9]|IE\d{7}[1-9][1-9]?"), None),
              (re.compile(r"[1-9]\d{10}"), None),
              (
                  re.compile(
                      r"\d{2}-\d{7}-\d|\d{11}|\d{2}-\d{9}-\d|\d{4}-\d{4}-\d{4}|\d{4}-\d{7}-\d"
                  ),
                  None,
              ),
              (
                  re.compile(r"\b\d{2}\s\d{3}\s\d{3}\s\d{3}\b|\b\d{11}\b"),
                  ("australian business number", "abn"),
              ),
      ],
      "id":[
              (
                  re.compile(
                      r"\d{6}([04][1-9]|[1256][0-9]|[37][01])(0[1-9]|1[0-2])\d{6}"
                  ),
                  None,
              )
      ],
      "es": [
              (re.compile(r"(?:ES)?\d{6-8}-?[A-Z]"), None),
              (
                  re.compile(r"\b[0-9]?[0-9]{7}[-]?[A-Z]\b"),
                  ("documento nacional de identidad", "DNI", "NIF", "identificación"),
              ),
              (re.compile(r"[1-9]\d?\d{6}|8\d{8}|9\d{8}|10\d{8}|11\d{8}|12\d{8}|"), None)
      ],
      "pt": [(re.compile(r"\d{3}\.d{3}\.d{3}-\d{2}|\d{11}"), None),
             (re.compile(r"PT\d{9}"), None),
      ],
      #from https://github.com/Aggregate-Intellect/bigscience_aisc_pii_detection/blob/main/language/zh/rules.py which is under Apache 2
      "zh": [
          (
              regex.compile(
                  r"(?:[16][1-5]|2[1-3]|3[1-7]|4[1-6]|5[0-4])\d{4}(?:19|20)\d{2}(?:(?:0[469]|11)(?:0[1-9]|[12][0-9]|30)|(?:0[13578]|1[02])(?:0[1-9]|[12][0-9]|3[01])|02(?:0[1-9]|[12][0-9]))\d{3}[\dXx]"
              ),
              None,
          ),
          (
              regex.compile(
                  r"(^[EeKkGgDdSsPpHh]\d{8}$)|(^(([Ee][a-fA-F])|([DdSsPp][Ee])|([Kk][Jj])|([Mm][Aa])|(1[45]))\d{7}$)"
              ),
              None,
          ),
          (
              regex.compile(
                  r"((\d{4}(| )\d{4}(| )\d{4}$)|([a-zA-Z][1-2]{1}[0-9]{8})|([0-3]{1}\d{8}))"
              ),
              None,
          ),
          (
              regex.compile('^(?:[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领 A-Z]{1}[A-HJ-NP-Z]{1}(?:(?:[0-9]{5}[DF])|(?:[DF](?:[A-HJ-NP-Z0-9])[0-9]{4})))|(?:[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领 A-Z]{1}[A-Z]{1}[A-HJ-NP-Z0-9]{4}[A-HJ-NP-Z0-9 挂学警港澳]{1})$'),
              None
          ),
          (
              regex.compile('\b[A-Z]{3}-\d{4}\b'),
              None,
          ),
          (
              regex.compile(
                  r"(0?\d{2,4}-[1-9]\d{6,7})|({\+86|086}-| ?1[3-9]\d{9} , ([\+0]?86)?[\-\s]?1[3-9]\d{9})"
              ),
              None,
          ),
          (
              regex.compile(
                  r"((\d{4}(| )\d{4}(| )\d{4}$)|([a-zA-Z][1-2]{1}[0-9]{8})|([0-3]{1}\d{8}))((02|03|037|04|049|05|06|07|08|089|082|0826|0836|886 2|886 3|886 37|886 4|886 49|886 5|886 6|886 7|886 8|886 89|886 82|886 826|886 836|886 9|886-2|886-3|886-37|886-4|886-49|886-5|886-6|886-7|886-8|886-89|886-82|886-826|886-836)(| |-)\d{4}(| |-)\d{4}$)|((09|886 9|886-9)(| |-)\d{2}(|-)\d{2}(|-)\d{1}(|-)\d{3})"
              ),
              None,
          ),
      ],
      "default": [
              #https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py ssn
              (
                  re.compile(
                     '(?!000|666|333)0*(?:[0-6][0-9][0-9]|[0-7][0-6][0-9]|[0-7][0-7][0-2])[- ](?!00)[0-9]{2}[- ](?!0000)[0-9]{4}'
                  ),
                  None,
              ),
              # https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py phone with exts
              (
                  re.compile('((?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*(?:[2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|(?:[2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?(?:[2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?(?:[0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(?:\d+)?))', re.IGNORECASE),
                  None
              ),
              # phone
              (
                  re.compile('''((?:(?<![\d-])(?:\+?\d{1,3}[-.\s*]?)?(?:\(?\d{3}\)?[-.\s*]?)?\d{3}[-.\s*]?\d{4}(?![\d-]))|(?:(?<![\d-])(?:(?:\(\+?\d{2}\))|(?:\+?\d{2}))\s*\d{2}\s*\d{3}\s*\d{4}(?![\d-])))'''),
                  None,
              ),
              #email
              (re.compile("([a-z0-9!#$%&'*+\/=?^_`{|.}~-]+@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)", re.IGNORECASE), None),
              #credit card
              (re.compile('((?:(?:\\d{4}[- ]?){3}\\d{4}|\\d{15,16}))(?![\\d])'), None),
              #ip
              (re.compile('(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)', re.IGNORECASE), None),
              #ipv6
              (re.compile('((?:(?:\\d{4}[- ]?){3}\\d{4}|\\d{15,16}))(?![\\d])'), None),
              #icd code - see https://stackoverflow.com/questions/5590862/icd9-regex-pattern
              (re.compile('[A-TV-Z][0-9][A-Z0-9](\.[A-Z0-9]{1,4})'), None),
              # generic government id. consider a more complicated string with \w+ at the beginning or end
              (re.compile(r"\d{8}|\d{9}|\d{10}|\d{11}"), None),
      ],
    },
 }

def detect_ner_with_regex_and_context(sentence, src_lang, context_window=5, tag_type={'ID'}, ignore_stdnum_type={'isil', 'isbn', 'isan', 'imo', 'gs1_128', 'grid', 'figi', 'ean', 'casrn', }):
      """
      This function returns a list of 3 tuples, representing an NER detection for [(entity, start, end, tag), ...]
      NOTE: There may be overlaps
      """
      global regex_rulebase
      if src_lang in ("zh", "ko", "ja"):
          sentence_set = set(sentence.lower())
      else:
          sentence_set = set(sentence.lower().split(" "))
      idx = 0
      all_ner = []
      original_sentence = sentence
      for tag, regex_group in regex_rulebase.items():
          if tag not in tag_type: continue
          for regex_context in regex_group.get(src_lang, []) + regex_group.get("default", []):
              if True:
                  regex, context = regex_context
                  found_context = False
                  if context:
                      for c1 in context:
                        c1 = c1.lower()
                        for c2 in c1.split():
                          if c2 in sentence_set:
                              found_context = True
                              break
                        if found_context: break
                      if not found_context:
                          continue
                  for ent in regex.findall(sentence):
                      if not isinstance(ent, str) or not ent:
                          continue
                      if tag == 'ID':
                          stnum_type = id_2_stdnum_type(ent)
                          #print (stnum_type, any(a for a in stnum_type if a in ignore_stdnum_type))
                          #TODO: we couold do a rule where given the language, map to country, and if the country stdnum is matched, 
                          #we won't skip this ent even if it also matches an ignore_stdnum_type
                          if any(a for a in stnum_type if a in ignore_stdnum_type):
                            continue
                      sentence2 = original_sentence
                      delta = 0
                      while True:
                        if ent not in sentence2:
                          break
                        else:
                          i = sentence2.index(ent)
                          j = i + len(ent)
                          if found_context:
                              len_sentence = len(sentence2)
                              left = sentence2[max(0, i - context_window) : i].lower()
                              right = sentence2[j : min(len_sentence, j + context_window)].lower()
                              found_context = False
                              for c in context:
                                c = c.lower()
                                if c in left or c in right:
                                    found_context = True
                                    break
                              if not found_context:
                                sentence2 = sentence2[i+len(ent):]
                                continue
                          sentence2 = sentence2[i+len(ent):]
                          all_ner.append((ent, delta+i, delta+j, tag))
                          delta += j
                          idx += 1
      return all_ner
