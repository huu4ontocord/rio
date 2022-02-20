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
import dateparser
try:
  from postal.parser import parse_address
except:
  def parse_address(s): return {}
import sys, os
try:
  sys.path.append(os.path.abspath(os.path.dirname(__file__)))    
except:
  pass
from stopwords import stopwords
from country_2_lang import *
from pii_regexes_rulebase import regex_rulebase
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
    'isrc':  stdnum.isrc.validate,
    'issn':  stdnum.issn.validate,
    'lei':  stdnum.lei.validate,
    'mac':  stdnum.mac.validate,
    'meid':  stdnum.meid.validate,
    'vatin':  stdnum.vatin.validate,
}

#this is based on lang_2_country. must be changed if we change lang_2_country.
lang_2_stdnum = {'am': [],
    'ar': ['il.hp', 'il.idnr'],
    'ay': [],
    'az': [],
    'be': ['by.unp'],
    'bg': ['bg.vat', 'bg.egn', 'bg.pnf'],
    'bi': [],
    'bn': [],
    'bs': [],
    'ca': ['ad.nrt'],
    'ch': [],
    'cs': ['cz.dic', 'cz.rc'],
    'da': ['dk.cvr', 'dk.cpr'],
    'de': ['at.businessid',
      'at.tin',
      'at.vnr',
      'at.postleitzahl',
      'at.uid',
      'ch.ssn',
      'ch.vat',
      'ch.uid',
      'ch.esr',
      'de.vat',
      'de.handelsregisternummer',
      'de.wkn',
      'de.stnr',
      'de.idnr',
      'be.iban',
      'be.vat',
      'li.peid',
      'lu.tva'],
    'dv': [],
    'dz': [],
    'el': ['gr.vat', 'gr.amka', 'cy.vat'],
    'en': ['au.abn',
      'au.acn',
      'au.tfn',
      'ca.sin',
      'ca.bn',
      'gb.upn',
      'gb.vat',
      'gb.nhs',
      'gb.utr',
      'gb.sedol',
      'ie.vat',
      'ie.pps',
      'in_.epic',
      'in_.gstin',
      'in_.pan',
      'in_.aadhaar',
      'nz.bankaccount',
      'nz.ird',
      'us.ssn',
      'us.atin',
      'us.rtn',
      'us.tin',
      'us.ein',
      'us.itin',
      'us.ptin',
      'mt.vat',
      'mu.nid',
      'sg.uen',
      'za.tin',
      'za.idnr'],
    'es': ['ar.dni',
      'ar.cbu',
      'ar.cuit',
      'es.iban',
      'es.ccc',
      'es.cif',
      'es.dni',
      'es.cups',
      'es.referenciacatastral',
      'es.nie',
      'es.nif',
      'mx.curp',
      'mx.rfc',
      'cl.rut',
      'co.nit',
      'cr.cpj',
      'cr.cr',
      'cr.cpf',
      'cu.ni',
      'do.rnc',
      'do.ncf',
      'do.cedula',
      'ec.ruc',
      'ec.ci',
      'gt.nit',
      'pe.cui',
      'pe.ruc',
      'py.ruc',
      'sv.nit',
      'uy.rut',
      've.rif'],
    'et': ['ee.ik', 'ee.registrikood', 'ee.kmkr'],
    'fa': [],
    'fi': ['fi.associationid',
      'fi.veronumero',
      'fi.hetu',
      'fi.ytunnus',
      'fi.alv'],
    'fil': [],
    'fj': [],
    'fo': [],
    'fr': ['ca.sin',
      'ca.bn',
      'ch.ssn',
      'ch.vat',
      'ch.uid',
      'ch.esr',
      'fr.siret',
      'fr.tva',
      'fr.nir',
      'fr.nif',
      'fr.siren',
      'be.iban',
      'be.vat',
      'lu.tva',
      'mc.tva',
      'mu.nid'],
    'ga': ['ie.vat', 'ie.pps'],
    'gil': [],
    'gn': ['py.ruc'],
    'gsw': ['ch.ssn', 'ch.vat', 'ch.uid', 'ch.esr', 'li.peid'],
    'gv': [],
    'he': ['il.hp', 'il.idnr'],
    'hi': ['in_.epic', 'in_.gstin', 'in_.pan', 'in_.aadhaar'],
    'hif': [],
    'ho': [],
    'hr': ['hr.oib'],
    'ht': [],
    'hu': ['hu.anum'],
    'hy': [],
    'id': ['id.npwp'],
    'is': ['is_.kennitala', 'is_.vsk'],
    'it': ['ch.ssn',
      'ch.vat',
      'ch.uid',
      'ch.esr',
      'it.codicefiscale',
      'it.aic',
      'it.iva',
      'sm.coe'],
    'ja': ['jp.cn'],
    'ka': [],
    'kk': [],
    'kl': [],
    'km': [],
    'ko': ['kr.rrn', 'kr.brn'],
    'ky': [],
    'lb': ['lu.tva'],
    'lo': [],
    'lt': ['lt.pvm', 'lt.asmens'],
    'lv': ['lv.pvn'],
    'mg': [],
    'mh': [],
    'mi': ['nz.bankaccount', 'nz.ird'],
    'mk': [],
    'mn': [],
    'ms': ['my.nric', 'sg.uen'],
    'mt': ['mt.vat'],
    'my': [],
    'na': [],
    'nb': ['no.mva', 'no.iban', 'no.kontonr', 'no.fodselsnummer', 'no.orgnr'],
    'nd': [],
    'ne': [],
    'niu': [],
    'nl': ['nl.bsn',
      'nl.brin',
      'nl.onderwijsnummer',
      'nl.btw',
      'nl.postcode',
      'be.iban',
      'be.vat'],
    'nn': ['no.mva', 'no.iban', 'no.kontonr', 'no.fodselsnummer', 'no.orgnr'],
    'ny': [],
    'pap': [],
    'pau': [],
    'pl': ['pl.pesel', 'pl.nip', 'pl.regon'],
    'ps': [],
    'pt': ['br.cnpj', 'br.cpf', 'pt.cc', 'pt.nif'],
    'qu': ['ec.ruc', 'ec.ci', 'pe.cui', 'pe.ruc'],
    'rn': [],
    'ro': ['ro.onrc', 'ro.cui', 'ro.cf', 'ro.cnp', 'md.idno'],
    'ru': ['ru.inn', 'ua.rntrc', 'ua.edrpou', 'by.unp'],
    'rw': [],
    'sg': [],
    'si': [],
    'sk': ['sk.rc', 'sk.dph'],
    'sl': ['si.ddv'],
    'sm': [],
    'sn': [],
    'so': [],
    'sq': ['al.nipt'],
    'sr': ['me.iban', 'rs.pib'],
    'ss': [],
    'st': [],
    'sv': ['fi.associationid',
      'fi.veronumero',
      'fi.hetu',
      'fi.ytunnus',
      'fi.alv',
      'se.vat',
      'se.personnummer',
      'se.postnummer',
      'se.orgnr'],
    'sw': [],
    'ta': ['sg.uen'],
    'tet': [],
    'tg': [],
    'th': ['th.pin', 'th.tin', 'th.moa'],
    'ti': [],
    'tk': [],
    'tkl': [],
    'tn': [],
    'to': [],
    'tpi': [],
    'tr': ['tr.vkn', 'tr.tckimlik', 'cy.vat'],
    'tvl': [],
    'ty': [],
    'tzm': [],
    'uk': ['ua.rntrc', 'ua.edrpou'],
    'ur': [],
    'uz': [],
    'vi': ['vn.mst'],
    'wni': [],
    'wo': [],
    'yo': [],
    'zdj': [],
    'zh': ['cn.ric', 'cn.uscc', 'tw.ubn', 'sg.uen'],
    'default':['bic',
      'bitcoin',
      'casrn',
      'cusip',
      'ean',
      'figi',
      'grid',
      'gs1_128',
      'iban',
      'imei',
      'imo',
      'imsi',
      'isan',
      'isbn',
      'isil',
      'isin',
      'ismn',
      'iso11649',
      'iso6346',
      'isrc',
      'issn',
      'lei',
      'mac',
      'meid',
      'vatin']
}

def ent_2_stdnum_type(text, src_lang=None):
  """ given a entity mention and the src_lang, determine potentially stdnum type """
  stdnum_type = []
  if src_lang is None:
    items = list(stdnum_mapper.items())
  else:
    l1 =  lang_2_stdnum.get(src_lang, []) + lang_2_stdnum.get('default', [])
    items = [(a1, stdnum_mapper[a1]) for a1 in l1]

  for ent_type, validate in items:
    try:
      found = validate(text)
    except:
      found = False
    if found:
      stdnum_type.append (ent_type)
  return stdnum_type

lstrip_chars = " ,،、<>{}[]|()\"'“”《》«»:;"
rstrip_chars = " ,،、<>{}[]|()\"'“”《》«»!:;?。.…．"
date_parser_lang_mapper = {'st': 'en', 'ny': 'en', 'xh': 'en'}


def test_is_date(ent, tag, sentence, len_sentence, is_cjk, i, src_lang, sw, year_start=1600, year_end=2050):
    """
    Helper function used to test if an ent is a date or not
    We use dateparse to find context words around the ID/date to determine if its a date or not.
    For example, 100 AD is a date, but 100 might not be.
    Input:
      :ent: an entity mention
      :tag: either ID or DATE
      :sentence: the context
      :is_cjk: if this is a Zh, Ja, Ko text
      :i: the position of ent in the sentence
     Returns:
        (ent, tag): potentially expanded ent, and the proper tag. 
        Could return a potentially expanded ent, and the proper tag. 
        Returns ent as None, if originally tagged as 'DATE' and it's not a DATE and we don't know what it is.
     
    """
    # perform some fast heuristics so we don't have to do dateparser
    len_ent = len(ent)
    if len_ent > 17 or (len_ent > 8 and to_int(ent)):
      if tag == 'DATE': 
        #this is a very long number and not a date
        return None, tag
      else:
        #no need to check the date
        return ent, tag 
        
    if not is_cjk:
      if i > 0 and sentence[i-1] not in lstrip_chars: 
        if tag == 'DATE': 
          return None, tag
        else:
          return ent, tag
      if i+len_ent < len_sentence - 1 and sentence[i+len_ent+1] not in rstrip_chars: 
        if tag == 'DATE': 
          return None, tag
        else:
          return ent, tag

    int_arr = [(e, to_int(e)) for e in ent.replace("/", "-").replace(" ","-").replace(".","-").split("-")]
    if is_fast_date(ent, int_arr): 
      #this is most likely a date
      return ent, 'DATE'

    for e, val in int_arr:
      if val is not None and len(e) > 8:
        if tag == 'DATE': 
          #this is a very long number and not a date
          return None, tag

    #test if this is a 4 digit year. we need to confirm it's a real date
    is_date = False
    is_4_digit_year = False
    if tag == 'DATE' and len_ent == 4:
      e = to_int(ent)
      is_4_digit_year = (e <= year_end and e >= year_start)
    
    #now do dateparser
    if not is_4_digit_year:
      is_date =  dateparser.parse(ent, languages=[date_parser_lang_mapper.get(src_lang,src_lang)]) # use src_lang to make it faster, languages=[src_lang])
    
    if (not is_date and tag == 'DATE') or (is_date and tag == 'ID'):
        j = i + len_ent
        #for speed we can just use these 6 windows to check for a date.
        #but for completeness we could check a sliding window. 
        #Maybe in some countries a year could
        #be in the middle of a date: Month Year Day
        ent_spans = [(-3,0), (-2, 0), (-1, 0), \
              (0, 3), (0, 2), (0, 1)]
        before = sentence[:i]
        after = sentence[j:]
        if before and not is_cjk and before[-1] not in lstrip_chars:
          is_date = False
        elif after and not is_cjk and after[0] not in rstrip_chars:
          is_date = False
        else:
          if  not is_cjk:
            before = before.split()
            after = after.split()
          len_after = len(after)
          len_before = len(before)
          for before_words, after_words in ent_spans:
            if after_words > len_after: continue
            if -before_words > len_before: continue 
            if before_words == 0: 
                before1 = []
            else:
                before1 = before[max(-len_before,before_words):]
            after1 = after[:min(len_after,after_words)]
            if is_cjk:
              ent2 = "".join(before1)+ent+"".join(after1)
            else:
              ent2 = " ".join(before1)+" "+ent+" "+" ".join(after1)
            if ent2.strip() == ent: continue
            is_date = dateparser.parse(ent2, languages=[date_parser_lang_mapper.get(src_lang,src_lang)])# use src_lang to make it faster, languages=[src_lang])
            if is_date:
              #sometimes dateparser says things like "in 2020" is a date, which it is
              #but we want to strip out the stopwords.
              if before1 and before1[-1].lower() in sw:
                before1 = before1[:-1]
              if after1 and after1[0].lower() in sw:
                after1 = after1[1:]
              if is_cjk:
                ent2 = "".join(before1)+ent+"".join(after1)
              else:
                ent2 = " ".join(before1)+" "+ent+" "+" ".join(after1)
              ent = ent2.strip()
              tag = "DATE"
              return ent, tag

    if tag == 'DATE' and not is_date:
      return None, tag

    return ent, tag

def to_int(s):
  try:
    return int(s)
  except:
    return None

def is_fast_date(ent, int_arr=None, year_start=1600, year_end=2050):
  """search for patterns like, yyyy-mm-dd, dd-mm-yyyy, yyyy-yyyy """
  if int_arr:
    len_int_arr = len(int_arr)
    if len_int_arr == 1 or len_int_arr > 3: return False
  if int_arr is None:
    ent_arr = ent.replace("/", "-").replace(" ","-").replace(".","-")
    if not ("-" in ent_arr and ent_arr.count("-") <=2): return False
    int_arr = [(e, to_int(e)) for e in ent_arr.split("-")]
  is_date = False
  has_year = has_month = has_day = 0
  for e, val in int_arr:
    if val is None: 
      break
    if (val <= year_end and val >= year_start):
      has_year +=1
    elif val <= 12 and val >= 1:
      has_month += 1
    elif val <= 31 and val >= 1:
      has_day += 1
    else:
      return False
  if (has_year == 1 and has_month == 1) or \
        (has_year == 2 and has_month == 0 and has_day == 0) or \
        (has_year == 1 and has_month == 1 and has_day == 1):
      return True
  return False

#cusip number probaly PII?
def detect_ner_with_regex_and_context(sentence, src_lang,  tag_type={'ID'}, prioritize_lang_match_over_ignore=True, \
      ignore_stdnum_type={'isil', 'isbn', 'isan', 'imo', 'gs1_128', 'grid', 'figi', 'ean', 'casrn', 'cusip' }, \
      all_regex=None, context_window=20, min_id_length=6, max_id_length=50, \
      precedence={'ID':0, 'PHONE':1, 'IP_ADDRESS':2, 'DATE':3, 'TIME':4, 'LICENSE_PLATE':5, 'USER':6, 'AGE':7, 'ADDRESS':8, 'URL':9}):
      """
      Output:
       - This function returns a list of 4 tuples, representing an NER detection for [(entity, start, end, tag), ...]
      Input:
       :sentence: the sentence to tag
       :src_lang: the language of the sentence
       :context_window: the contxt window in characters to check for context characters for any rules that requries context
       :max_id_length: the maximum length of an ID
       :min_id_length: the minimum length of an ID
       :tag_type: the type of NER tags we are detecting. If None, then detect everything.
       :ignore_stdnum_type: the set of stdnum we will consider NOT PII and not match as an ID
       :prioritize_lang_match_over_ignore: if true, and an ID matches an ingore list, we still keep it as an ID if there was an ID match for this particular src_lang
       :all_regex: a rulebase of the form {tag: {lang: [(regex, context, block), ...], 'default': [(regex, context, block), ...]}}. 
         context are words that must be found surronding the entity. block are words that must not be found.
         If all_regex is none, then we use the global regex_rulebase
       
      ALGORITHM:
        For each regex, we check the sentence to find a match and a required context, if the context exists in a window.
        If the regex is an ID or a DATE, test to see if it's a stdnum we know. Stdnum are numbers formatted to specific regions, or generally.
        If it is a stdnum and NOT a PII type (such as ISBN numbers) skip this ID.
          UNLESS If the stdnum is ALSO a PII type for the local region of the language, then consider it a matched stdnum.
        If it's a matched stdnum that is not skipped, save it as an ID.
        If the ID is not a stdum, check if the ID is a DATE. If it's a DATE using context words in a context window. 
          If it's a DATE then save it as a DATE, else save as ID.
        Gather all regex matches and sort the list by position, prefering longer matches, and DATEs and ADDRESSES over IDs.
        For all subsumed IDs and DATEs, remove those subsumed items. 
        Return a list of potentially overlapping NER matched.
      NOTE: 
      - There may be overlaps in mention spans. 
      - Unlike presidio, we require that a context be met. We don't increase a score if a context is matched.  
      - A regex does not need to match string boundaries or space boundaries. The matching code checks this. 
          We require all entities that is not cjk to have space or special char boundaries or boundaries at end or begining of sentence.
      - As such, We don't match embedded IDs: e.g., MyIDis555-555-5555 won't match the ID. This is to preven
        matching extremely nosiy imput that might have patterns of numbers in long strings.
      
      """

      sw = stopwords.get(src_lang, {})
      
      # if we are doing 'ID', we would still want to see if we catch an ADDRESS. 
      # ADDRESS may have higher precedence, in which case it might overide an ID match. 
      no_address = False
      if tag_type is not None and 'ID' in tag_type and 'ADDRESS' not in tag_type:
         no_address = True
         tag_type = set(list(tag_type)+['ADDRESS'])
         
      # if we are doing 'DATE' we would still want to do ID because they intersect.
      no_id = False
      if tag_type is not None and 'DATE' in tag_type and 'ID' not in tag_type:
         no_id = True
         tag_type = set(list(tag_type)+['ID'])
        
      is_cjk = src_lang in ("zh", "ko", "ja")
      if is_cjk:
          sentence_set = set(sentence.lower())
      else:
          sentence_set = []
          #let's do a sanity check. there should be no words beyond 100 chars.
          #this will really mess up our regexes.
          for word in sentence.split(" "):
            len_word = len(word)
            if len_word > 100:
              sentence = sentence.replace(word, " "*len_word)
            else:
              sentence_set.append(word.lower())
          sentence_set = set(sentence_set)
      all_ner = []
      len_sentence = len(sentence)
        
      if all_regex is None:
        all_regex = regex_rulebase
      if tag_type is None:
        all_tags_to_check = list(all_regex.keys())
      else:
        all_tags_to_check = list(tag_type) 

      for tag in all_tags_to_check:
          regex_group = all_regex.get(tag)
          if not regex_group: continue
          for regex_context, extra_weight in [(a, 1) for a in regex_group.get(src_lang, [])] + [(a, 0) for a in regex_group.get("default", [])]:
              if True:
                  regex, context, block = regex_context
                  #if this regex rule requires a context, find if it is satisified in general. this is a quick check.
                  potential_context = False
                  if context:
                      for c1 in context:
                        c1 = c1.lower()
                        for c2 in c1.split():
                          if c2 in sentence_set:
                              potential_context = True
                              break
                        if potential_context: break
                      if not potential_context:
                          continue

                  #now apply regex
                  for ent in list(set(list(regex.findall(sentence)))):
                      
                      if not isinstance(ent, str):
                        continue
                      ent = ent.strip()
                      #ent = ent.rstrip(rstrip_chars)
                      #ent = ent.lstrip(lstrip_chars)
                      if not ent:
                        continue
 
                      ent_is_4_digit=False
                      len_ent = len(ent)
                      if len_ent == 4:
                        try:
                          int(ent)
                          ent_is_4_digit=True
                        except:
                          ent_is_4_digit=False
                      sentence2 = sentence
                      delta = 0
                      #check to see if the ID or DATE is type of stdnum
                      is_stdnum = False
                      if tag in ('ID', 'DATE'):
                          #simple length test
                          ent_no_space = ent.replace(" ", "").replace(".", "").replace("-", "")
                          if len(ent_no_space) > max_id_length and tag == 'ID': continue
                          if len(ent_no_space) < min_id_length and tag == 'ID': continue
                            
                          #check if this is really a non PII stdnum, unless it's specifically an ID for a country using this src_lang. 
                          #TODO - complete the country to src_lang dict above. 
                          stnum_type = ent_2_stdnum_type(ent, src_lang)
                          
                          #if the stdnum is one of the non PII types, we will ignore it
                          if prioritize_lang_match_over_ignore:
                                is_stdnum = any(a for a in stnum_type if "." in a and src_lang in country_2_lang.get(a.split(".")[0], []))
                          if not ent_is_4_digit and not is_stdnum and any(a for a in stnum_type if a in ignore_stdnum_type):
                            #a four digit entity might be a year, so don't skip this ent
                            continue
                          #this is actually an ID of known stdnum and not a DATE
                          if any(a for a in stnum_type if a not in ignore_stdnum_type):
                            tag = 'ID'
                            is_stdnum = True

                      #let's check the FIRST instance of this DATE or ID is really a date; 
                      #ideally we should do this for every instance of this ID
                      if tag == 'DATE' or (tag == 'ID' and not is_stdnum):
                        ent, tag = test_is_date(ent, tag, sentence, len_sentence, is_cjk, sentence.index(ent),  src_lang, sw)
                        if not ent: continue
                      
                      #now let's turn all occurances of ent in this sentence into a span mention and also check for context and block words
                      len_ent = len(ent)
                      while True:
                        if ent not in sentence2:
                          break
                        else:
                          i = sentence2.index(ent)
                          j = i + len_ent
                          if potential_context or block:
                              len_sentence2 = len(sentence2)
                              left = sentence2[max(0, i - context_window) : i].lower()
                              right = sentence2[j : min(len_sentence2, j + context_window)].lower()
                              found_context = False
                              if context:
                                for c in context:
                                  c = c.lower()
                                  if c in left or c in right:
                                      found_context = True
                                      break
                              else:
                                found_context = True
                              if block:
                                for c in block:
                                  c = c.lower()
                                  if c in left or c in right:
                                      found_context = False
                                      break
                              if not found_context:
                                delta += j
                                sentence2 = sentence2[i+len(ent):]
                                continue
                          #check to see if the entity is really a standalone word or part of another longer word.
                          # for example, we wont match a partial set of very long numbers as a 7 digit ID for example
                          if is_cjk or ((i+delta == 0 or sentence2[i-1]  in lstrip_chars) and (j+delta >= len_sentence-1 or sentence2[j] in rstrip_chars)): 
                            all_ner.append((ent, delta+i, delta+j, tag, extra_weight))
                          sentence2 = sentence2[i+len(ent):]
                          delta += j
                            
      all_ner = list(set(all_ner))
      # sort by length and position, favoring non-IDs first using the precedence list, 
      # and additionaly giving one extra weight to language specific regex (as opposed to default rules).
      # this doesn't do a perfect overlap match; just an overlap to the prior item.
      all_ner.sort(key=lambda a: a[1]+(1.0/(1.0+(100*((precedence.get(a[3], min(20,len(a[3])))+a[4]))+a[2]-a[1]))))
      #print (all_ner)
      if not tag_type or 'ID' in tag_type:
        # now do overlaps prefering longer ents, and higher prededence items over embedded IDs or dates, etc.
        all_ner2 = []
        prev_mention = None
        for mention in all_ner:
          if prev_mention:
            if (prev_mention[1] == mention[1] and prev_mention[3] == mention[3] and prev_mention[4] != mention[4]) or\
              (prev_mention[2] >= mention[1] and prev_mention[2] >= mention[2]): 
              #either a shoter lang specific mention takes predence or a subsuming mention takes precedence
              #and prev_mention[3] in ('ID', 'DATE', 'ADDRESS') and 
              # if there is any complete overlap, then we use the precedence rules
              # an alternate: if there is a complete overlap to an ID in an ADDRESS or a DATE, we ignore this ID
              #               this is because we have more context for the DATE or ADDRESS to determine it is so. 
              #if mention[3] in ('DATE', 'ID', 'PHONE'): 
                continue
            else:
              prev_mention = mention
          else:
            prev_mention = mention
          all_ner2.append(mention[:4])
        all_ner = all_ner2
      if no_address:
         all_ner = [a for a in all_ner if a[3] != 'ADDRESS']
      if no_id:
         all_ner = [a for a in all_ner if a[3] != 'ID']   
        
      return all_ner
