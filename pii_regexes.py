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

#This is an incomplete list. TODO - adapt from https://github.com/scrapinghub/dateparser/blob/master/dateparser/data/languages_info.py
country_2_lang = {'pk': 'ur',
     'ar': 'es',
     'ae': 'ar',
     'iq': 'ar',
     'dz': 'ar',
     'eg': 'ar',
     'sd': 'ar',
     'au': 'en',
     'ad': 'ca',
     'aa': 'ar',
     'am': 'hy',
     'at': 'de',
     'bg': 'bg',
     'br': 'pt',
     'ca': 'fr',
     'ch': 'fr',
     'cn': 'zh',
     'cz': 'cs',
     'de': 'de',
     'dk': 'dk',
     'ee': 'et',
     'es': 'es',
     'fi': 'fi',
     'fr': 'fr',
     'gb': 'en',
     'ge': 'ka',
     'gh': 'tw',
     'gr': 'el',
     'hr': 'hr',
     'hu': 'hu',
     'id': 'id',
     'ie': 'ga',
     'il': 'he',
     'in_': 'ta',
     'ir': 'fa',
     'it': 'it',
     'jp': 'ja',
     'kr': 'ko',
     'lt': 'lt',
     'lv': 'lv',
     'mx': 'es',
     'nl': 'nl',
     'no': 'no',
     'np': 'ne',
     'nz': 'en',
     'pl': 'pl',
     'ps': 'ar',
     'pt': 'pt',
     'qc': 'fr',
     'ro': 'ro',
     'ru': 'ru',
     'sa': 'ar',
     'se': 'sv',
     'si': 'sl',
     'th': 'th',
     'tr': 'tr',
     'tw': 'zh',
     'ua': 'uk',
     'us': 'en'}

#TODO - get the complete list for our language
lang_2_country = {   
    'am': ['et'],
    'ar': [   
        'ae','iq','dz','eg','sd','aa','il','ps','sa','bh','km','dj','er','eh',
        'jo','kw','lb','ly','ma','mr','om','qa','so','sy','td','tn','ye'
        ],
    'ay': ['bo'],
    'az': ['az'],
    'be': ['by'],
    'bg': ['bg'],
    'bi': ['vu'],
    'bn': ['bd'],
    'bs': ['ba'],
    'ca': ['ad'],
    'ch': ['gu'],
    'cs': ['cz'],
    'da': ['dk'],
    'de': ['at', 'ch', 'de', 'be', 'li', 'lu'],
    'dv': ['mv'],
    'dz': ['bt'],
    'el': ['gr', 'cy'],
    'en': [  
        'pk','sd','au','ca','gb','gh','ie','in','us','ai','ag','bi','bs','bz',
        'bm','bb','bw','cm','ck','cx','ky','dm','er','fj','fk','fm','gg','gi',
        'gm','gd','gy','hk','im','io','jm','je','ke','ki','kn','lr','lc','ls',
        'mg','mh','mt','ms','mu','mw','na','nf','ng','nu','nr','pn','ph','pw',
        'pg','rw','sg','sh','sb','sl','ss','sz','sx','sc','tc','tk','to','tt',
        'tv','tz','ug','vc','vg','vu','ws','za','zm','zw'
        ],
    'es': [   
        'ar','es','mx','bo','cl','co','cr','cu','do','ec','gq','gt','hn','ni',
        'pa','pe','pr','py','sv','uy','ve'
        ],
    'et': ['ee'],
    'fa': ['ir', 'af'],
    'fi': ['fi'],
    'fil': ['ph'],
    'fj': ['fj'],
    'fo': ['fo'],
    'fr': [
        'dz', 'ca', 'ch', 'fr', 'qc', 'bi', 'be', 'bj', 'bf', 'bl', 'cf', 'ci',
        'cm', 'cd', 'cg', 'km', 'dj', 'ga', 'gn', 'gp', 'gq', 'gf', 'ht', 'lu', 
        'mf', 'mc', 'mg', 'ml', 'mq', 'mu', 'yt', 'nc', 'ne', 'pf', 're', 'rw', 
        'sn', 'pm', 'sc', 'sy', 'td', 'tg', 'tn', 'vu', 'wf'
    ],
    'ga': ['ie'],
    'gil': ['ki'],
    'gn': ['py'],
    'gv': ['im'],
    'he': ['il'],
    'hi': ['in'],
    'hif': ['fj'],
    'ho': ['pg'],
    'hr': ['hr', 'ba'],
    'ht': ['ht'],
    'hu': ['hu'],
    'hy': ['am'],
    'id': ['id'],
    'is': ['is'],
    'it': ['ch', 'it', 'sm'],
    'ja': ['jp'],
    'ka': ['ge'],
    'kk': ['kz'],
    'kl': ['gl'],
    'km': ['kh'],
    'ko': ['kr', 'kp'],
    'ky': ['kg'],
    'lb': ['lu'],
    'lo': ['la'],
    'lt': ['lt'],
    'lv': ['lv'],
    'mg': ['mg'],
    'mh': ['mh'],
    'mi': ['nz'],
    'mk': ['mk'],
    'mn': ['mn'],
    'ms': ['bn', 'my', 'sg'],
    'mt': ['mt'],
    'my': ['mm'],
    'na': ['nr'],
    'nb': ['no', 'sj'],
    'nd': ['zw'],
    'ne': ['np'],
    'niu': ['nu'],
    'nl': ['nl', 'aw', 'be', 'bq', 'cw', 'sr', 'sx'],
    'nn': ['no'],
    'ny': ['mw'],
    'pap': ['aw'],
    'pau': ['pw'],
    'pl': ['pl'],
    'ps': ['af'],
    'pt': ['br', 'pt', 'ao', 'cv', 'gw', 'gq', 'mo', 'mz', 'st', 'tl'],
    'qu': ['bo', 'ec', 'pe'],
    'rn': ['bi'],
    'ro': ['ro', 'md'],
    'ru': ['ru', 'by', 'kz', 'kg'],
    'rw': ['rw'],
    'sg': ['cf'],
    'si': ['lk'],
    'sk': ['sk'],
    'sl': ['si'],
    'sm': ['as', 'ws'],
    'sn': ['zw'],
    'so': ['so'],
    'sq': ['al'],
    'sr': ['ba', 'me', 'rs'],
    'ss': ['sz'],
    'st': ['ls'],
    'sv': ['fi', 'se', 'ax'],
    'sw': ['ke', 'tz', 'ug'],
    'ta': ['lk', 'sg'],
    'tet': ['tl'],
    'tg': ['tj'],
    'th': ['th'],
    'tk': ['tm'],
    'tkl': ['tk'],
    'tn': ['bw'],
    'to': ['to'],
    'tpi': ['pg'],
    'tr': ['tr', 'cy'],
    'tvl': ['tv'],
    'ty': ['pf'],
    'tzm': ['ma'],
    'uk': ['ua'],
    'ur': ['pk'],
    'uz': ['uz'],
    'vi': ['vn'],
    'wni': ['km'],
    'yo': ['ng'],
    'zdj': ['km'],
    'zh': ['cn', 'tw', 'hk', 'mo', 'sg']
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
    'isrc':  stdnum.isrc.validate,
    'issn':  stdnum.issn.validate,
    'lei':  stdnum.lei.validate,
    'mac':  stdnum.mac.validate,
    'meid':  stdnum.meid.validate,
    'vatin':  stdnum.vatin.validate,
}

#TODO - complete this list
lang_2_stdnum = {'ar': {},
 'bg': {'bg.egn': stdnum.bg.egn.validate,
  'bg.pnf': stdnum.bg.pnf.validate,
  'bg.vat': stdnum.bg.vat.validate},
 'ca': {'ad.nrt': stdnum.ad.nrt.validate},
 'cs': {'cz.dic': stdnum.cz.dic.validate,
  'cz.rc': stdnum.cz.rc.validate},
 'de': {'at.businessid': stdnum.at.businessid.validate,
  'at.postleitzahl': stdnum.at.postleitzahl.validate,
  'at.tin': stdnum.at.tin.validate,
  'at.uid': stdnum.at.uid.validate,
  'at.vnr': stdnum.at.vnr.validate,
  'de.handelsregisternummer': stdnum.de.handelsregisternummer.validate,
  'de.idnr': stdnum.de.idnr.validate,
  'de.stnr': stdnum.de.stnr.validate,
  'de.vat': stdnum.de.vat.validate,
  'de.wkn': stdnum.de.wkn.validate},
 'dk': {'dk.cpr': stdnum.dk.cpr.validate,
  'dk.cvr': stdnum.dk.cvr.validate},
 'el': {'gr.amka': stdnum.gr.amka.validate,
  'gr.vat': stdnum.gr.vat.validate},
 'en': {'au.abn': stdnum.au.abn.validate,
  'au.acn': stdnum.au.acn.validate,
  'au.tfn': stdnum.au.tfn.validate,
  'gb.nhs': stdnum.gb.nhs.validate,
  'gb.sedol': stdnum.gb.sedol.validate,
  'gb.upn': stdnum.gb.upn.validate,
  'gb.utr': stdnum.gb.utr.validate,
  'gb.vat': stdnum.gb.vat.validate,
  'nz.bankaccount': stdnum.nz.bankaccount.validate,
  'nz.ird': stdnum.nz.ird.validate,
  'us.atin': stdnum.us.atin.validate,
  'us.ein': stdnum.us.ein.validate,
  'us.itin': stdnum.us.itin.validate,
  'us.ptin': stdnum.us.ptin.validate,
  'us.rtn': stdnum.us.rtn.validate,
  'us.ssn': stdnum.us.ssn.validate,
  'us.tin': stdnum.us.tin.validate},
 'es': {'ar.cbu': stdnum.ar.cbu.validate,
  'ar.cuit': stdnum.ar.cuit.validate,
  'ar.dni': stdnum.ar.dni.validate,
  'es.ccc': stdnum.es.ccc.validate,
  'es.cif': stdnum.es.cif.validate,
  'es.cups': stdnum.es.cups.validate,
  'es.dni': stdnum.es.dni.validate,
  'es.iban': stdnum.es.iban.validate,
  'es.nie': stdnum.es.nie.validate,
  'es.nif': stdnum.es.nif.validate,
  'es.referenciacatastral': stdnum.es.referenciacatastral.validate,
  'mx.curp': stdnum.mx.curp.validate,
  'mx.rfc': stdnum.mx.rfc.validate},
 'et': {'ee.ik': stdnum.ee.ik.validate,
  'ee.kmkr': stdnum.ee.kmkr.validate,
  'ee.registrikood': stdnum.ee.registrikood.validate},
 'fa': {},
 'fi': {'fi.alv': stdnum.fi.alv.validate,
  'fi.associationid': stdnum.fi.associationid.validate,
  'fi.hetu': stdnum.fi.hetu.validate,
  'fi.veronumero': stdnum.fi.veronumero.validate,
  'fi.ytunnus': stdnum.fi.ytunnus.validate},
 'fr': {'ca.bn': stdnum.ca.bn.validate,
  'ca.sin': stdnum.ca.sin.validate,
  'ch.esr': stdnum.ch.esr.validate,
  'ch.ssn': stdnum.ch.ssn.validate,
  'ch.uid': stdnum.ch.uid.validate,
  'ch.vat': stdnum.ch.vat.validate,
  'fr.nif': stdnum.fr.nif.validate,
  'fr.nir': stdnum.fr.nir.validate,
  'fr.siren': stdnum.fr.siren.validate,
  'fr.siret': stdnum.fr.siret.validate,
  'fr.tva': stdnum.fr.tva.validate},
 'ga': {'ie.pps': stdnum.ie.pps.validate,
  'ie.vat': stdnum.ie.vat.validate},
 'he': {'il.hp': stdnum.il.hp.validate,
  'il.idnr': stdnum.il.idnr.validate},
 'hr': {'hr.oib': stdnum.hr.oib.validate},
 'hu': {'hu.anum': stdnum.hu.anum.validate},
 'hy': {},
 'id': {'id.npwp': stdnum.id.npwp.validate},
 'it': {'it.aic': stdnum.it.aic.validate,
  'it.codicefiscale': stdnum.it.codicefiscale.validate,
  'it.iva': stdnum.it.iva.validate},
 'ja': {'jp.cn': stdnum.jp.cn.validate},
 'ka': {},
 'ko': {'kr.brn': stdnum.kr.brn.validate,
  'kr.rrn': stdnum.kr.rrn.validate},
 'lt': {'lt.asmens': stdnum.lt.asmens.validate,
  'lt.pvm': stdnum.lt.pvm.validate},
 'lv': {'lv.pvn': stdnum.lv.pvn.validate},
 'ne': {},
 'nl': {'nl.brin': stdnum.nl.brin.validate,
  'nl.bsn': stdnum.nl.bsn.validate,
  'nl.btw': stdnum.nl.btw.validate,
  'nl.onderwijsnummer': stdnum.nl.onderwijsnummer.validate,
  'nl.postcode': stdnum.nl.postcode.validate},
 'no': {'no.fodselsnummer': stdnum.no.fodselsnummer.validate,
  'no.iban': stdnum.no.iban.validate,
  'no.kontonr': stdnum.no.kontonr.validate,
  'no.mva': stdnum.no.mva.validate,
  'no.orgnr': stdnum.no.orgnr.validate},
 'pl': {'pl.nip': stdnum.pl.nip.validate,
  'pl.pesel': stdnum.pl.pesel.validate,
  'pl.regon': stdnum.pl.regon.validate},
 'pt': {'br.cnpj': stdnum.br.cnpj.validate,
  'br.cpf': stdnum.br.cpf.validate,
  'pt.cc': stdnum.pt.cc.validate,
  'pt.nif': stdnum.pt.nif.validate},
 'ro': {'ro.cf': stdnum.ro.cf.validate,
  'ro.cnp': stdnum.ro.cnp.validate,
  'ro.cui': stdnum.ro.cui.validate,
  'ro.onrc': stdnum.ro.onrc.validate},
 'ru': {'ru.inn': stdnum.ru.inn.validate},
 'sl': {'si.ddv': stdnum.si.ddv.validate},
 'sv': {'se.orgnr': stdnum.se.orgnr.validate,
  'se.personnummer': stdnum.se.personnummer.validate,
  'se.postnummer': stdnum.se.postnummer.validate,
  'se.vat': stdnum.se.vat.validate},
 'ta': {'in_.aadhaar': stdnum.in_.aadhaar.validate,
  'in_.epic': stdnum.in_.epic.validate,
  'in_.gstin': stdnum.in_.gstin.validate,
  'in_.pan': stdnum.in_.pan.validate},
 'th': {'th.moa': stdnum.th.moa.validate,
  'th.pin': stdnum.th.pin.validate,
  'th.tin': stdnum.th.tin.validate},
 'tr': {'tr.tckimlik': stdnum.tr.tckimlik.validate,
  'tr.vkn': stdnum.tr.vkn.validate},
 'tw': {},
 'uk': {'ua.edrpou': stdnum.ua.edrpou.validate,
  'ua.rntrc': stdnum.ua.rntrc.validate},
 'zh': {'cn.ric': stdnum.cn.ric.validate,
  'cn.uscc': stdnum.cn.uscc.validate,
  'tw.ubn': stdnum.tw.ubn.validate},
 'default': {
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
}


def ent_2_stdnum_type(text, src_lang=None):
  stdnum_type = []
  if src_lang is None:
    items = list(stdnum_mapper.items())
  else:
    items = list(lang_2_stdnum.get(src_lang, {}).items()) + list(lang_2_stdnum['default'].items())

  for ent_type, validate in items:
    try:
      found = validate(text)
    except:
      found = False
    if found:
      stdnum_type.append (ent_type)
  return stdnum_type


#from https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py which is under the MIT License
# see also for ICD https://stackoverflow.com/questions/5590862/icd9-regex-pattern - but this could be totally wrong!
# we do regex in this order in order to not capture ner inside domain names and email addresses.
#NORP, AGE, ADDRESS and DISEASE regexes are just test cases. We will use transformers and rules to detect these.
regex_rulebase_extended = {
    #https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py
    "DATE": {
        "default": [
            #year
            (re.compile('\d{4}'), None),
            #date
            (re.compile('(?:(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?\s+(?:of\s+)?(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)|(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)\s+(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?)(?:\,)?\s*(?:\d{4})?|[0-3]?\d[-\./][0-3]?\d[-\./]\d{2,4}', re.IGNORECASE), None),
        ],
    },
    #https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py
    "TIME": {
        "default": [(re.compile('\d{1,2}:\d{2} ?(?:[ap]\.?m\.?)?|\d[ap]\.?m\.?', re.IGNORECASE), None),],
    },
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
       "zh": [(regex.compile(r"\d{1,3}歲|\d{1,3}岁|[一二三四五六七八九十百]{1,3}岁|[一二三四五六七八九十百]{1,3}歲"), None)],
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
             #https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py
              (
                  re.compile(r"P\.? ?O\.? Box \d+"), None
              )
      ],
      #from https://github.com/Aggregate-Intellect/bigscience_aisc_pii_detection/blob/main/language/zh/rules.py which is under Apache 2
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
      "zh": [(regex.compile(r"(糖尿|癌症|抗癌|爱滋|艾滋|愛滋|阿茲海默|老人痴呆|老人癡呆|心臟病|心脏病)", re.IGNORECASE), None)]
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
          #consider whether we want to make PHONE a separate tag, that collapses to ID
          #phone
          (re.compile(r"\d{4}-\d{8}"), None),
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
              (re.compile(r"\d{6,13}|[૦-૯]{6,13}|[೦-೯]{6,13}|[൦-൯]{6,13}|[୦-୯]{6,13}|[௦-௯]{6,13}|[۰-۹]{6,13}|[০-৯]{6,13}|[٠-٩]{6,13}|[壹-玖〡-〩零〇-九十廿卅卌百千万亿兆]{6,13}"), None),
              #more generic ids
              (
                  re.compile(
                      r"\d{2}-\d{7}-\d|\d{11}|\d{2}-\d{9}-\d|\d{4}-\d{4}-\d{4}|\d{4}-\d{7}-\d"
                  ),
                  None,
              ),
              # generic id with dashes
              (re.compile('[A-Z]{0,3}(?:[- ]*\d){6,13}'), None),
              # generic user id
              (re.compile(r"\S*@[a-zA-Z]+\S*"), None),
              # bitcoin
              (re.compile('(?<![a-km-zA-HJ-NP-Z0-9])[13][a-km-zA-HJ-NP-Z0-9]{26,33}(?![a-km-zA-HJ-NP-Z0-9])'), None),
      ],
    },
 }

# Some of this code from https://github.com/bigscience-workshop/data_tooling/blob/master/ac_dc/anonymization.py which is under the Apache 2 license
regex_rulebase = {
    "AGE": {
      "en": [
          (
              re.compile(
                  r"\S+ years old|\S+\-years\-old|\S+ year old|\S+\-year\-old", re.IGNORECASE
              ),
              None, None
          )
      ],
       "zh": [(regex.compile(r"\d{1,3}歲|\d{1,3}岁"), None, None)],
    },
    #https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py
    "DATE": {
      "default": [
            #year
            (re.compile('\d{4}'), None, None),# [('died', 'AGE'), ('born', 'AGE')]),
            (re.compile(r"\d{4}-\d{4}"), None, None),#[('died', 'AGE'), ('born', 'AGE')]), # yyyy-yyyy
            (re.compile(r"\d{4}-\d{2}-\d{2}"), None, None),#[('died', 'AGE'), ('born', 'AGE')]),  # yyyy-mm-dd or yyyy-dd-mm
            (re.compile(r"\d{2}-\d{2}-\d{4}"), None, None),# [('died', 'AGE'), ('born', 'AGE')]),  # mm-dd-yyyy or dd-mm-yyyy
            (re.compile(r"\d{2}-\d{4}"), None, None),# [('died', 'AGE'), ('born', 'AGE')]),  # mm-yyyy
            (re.compile(r"\d{4}-\d{2}"), None, None),# [('died', 'AGE'), ('born', 'AGE')]),  # yyyy-mm
            ##date
            (re.compile(r'[0-3]?\d[-\./][0-3]?\d[-\./]\d{2,4}'), None, None),# [('died', 'AGE'), ('born', 'AGE')]),
            #(re.compile('(?:(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?\s+(?:of\s+)?(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)|(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)\s+(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?)(?:\,)?\s*(?:\d{4})?|[0-3]?\d[-\./][0-3]?\d[-\./]\d{2,4}', re.IGNORECASE), , [('died', 'AGE'), ('born', 'AGE')]),
        ],
    },
    #https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py
    "TIME": {
      "default": [(re.compile('\d{1,2}:\d{2} ?(?:[ap]\.?m\.?)?|\d[ap]\.?m\.?', re.IGNORECASE), None, None),],
    },
    "URL": {
      "default": [(re.compile('https?:\/\/[^\s\"\']{8,50}|www[^\s\"\']{8,50}'), None, None)],
    },
    "ADDRESS": {
      "en": [
              #https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py
              (
                  re.compile(
                      r"\d{1,4} [\w\s]{1,20} (?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|park|parkway|pkwy|circle|cir|boulevard|blvd)\W?(?=\s|$).*\b\d{5}(?:[-\s]\d{4})?\b|\d{1,4} [\w\s]{1,20} (?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|park|parkway|pkwy|circle|cir|boulevard|blvd)\W?(?=\s|$)", re.IGNORECASE
                  ),
                  None, None
              ),
             #https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py
              (
                  re.compile(r"P\.? ?O\.? Box \d+"), None, None
              )
      ],
      #from https://github.com/Aggregate-Intellect/bigscience_aisc_pii_detection/blob/main/language/zh/rules.py which is under Apache 2
      "zh": [
          (
              regex.compile(
                  r"((\p{Han}{1,3}(自治区|省))?\p{Han}{1,4}((?<!集)市|县|州)\p{Han}{1,10}[路|街|道|巷](\d{1,3}[弄|街|巷])?\d{1,4}号)"
              ),
              None, None
          ),
          (
              regex.compile(
                  r"(?<zipcode>(^\d{5}|^\d{3})?)(?<city>\D+[縣市])(?<district>\D+?(市區|鎮區|鎮市|[鄉鎮市區]))(?<others>.+)"
              ),
              None, None
          ),
      ],
    },
    "PHONE": {
      "zh" : [(re.compile(r"\d{4}-\d{8}"), None, None),],
      # we can probably remove one f the below
      "default": [
              # https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py phone with exts
              (
                  re.compile('((?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*(?:[2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|(?:[2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?(?:[2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?(?:[0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(?:\d+)?))', re.IGNORECASE),
                  None, None
              ),
              # common regex phone
              (
                  re.compile('((?:(?<![\d-])(?:\+?\d{1,3}[-.\s*]?)?(?:\(?\d{3}\)?[-.\s*]?)?\d{3}[-.\s*]?\d{4}(?![\d-]))|(?:(?<![\d-])(?:(?:\(\+?\d{2}\))|(?:\+?\d{2}))\s*\d{2}\s*\d{3}\s*\d{4}(?![\d-])))'),
                  None, None
              ), 
              ( re.compile('[\+\d]?(\d{2,3}[-\.\s]??\d{2,3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})'), None, None)     
      ]      
    },
    "IP_ADDRESS": {
        "default": [(re.compile('(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)', re.IGNORECASE), None, None),]
              
        },
    "USER": {
      "default": [
              # generic user id
              (re.compile(r"\s@[a-z][0-9a-z]{4-8}", re.IGNORECASE), None, None),
              #email
              (re.compile("(\w+[a-z0-9!#$%&'*+\/=?^_`{|.}~-]*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)", re.IGNORECASE), None, None),
      ]    
    },
    "ID": {
      "default": [
              #credit card from common regex
              (re.compile('((?:(?:\\d{4}[- ]?){3}\\d{4}|\\d{15,16}))(?![\\d])'), None, None),
              #icd code - see https://stackoverflow.com/questions/5590862/icd9-regex-pattern
              (re.compile('[A-TV-Z][0-9][A-Z0-9](\.[A-Z0-9]{1,4})'), None, None),
              # generic id with dashes
              (re.compile('[A-Z#]{0,3}(?:[- ]*\d){6,13}'), None, ('pp', 'pp.')),
              # IBAN
              (re.compile('[A-Z]{2}\d+\d+[A-Z]{0,4}(?:[- ]*\d){10,32}[A-Z]{0,3}'), None, None),
              #ES45 0075 3494 7706 0100 0263
      ],
    },
 }

lstrip_chars = " ,،、<>{}[]|()\"'“”《》«»:;"
rstrip_chars = " ,،、<>{}[]|()\"'“”《》«»!:;?。.…．"
date_parser_lang_mapper = {'st': 'en', 'ny': 'en', 'xh': 'en'}
from stopwords import stopwords
#cusip number probaly PII?
def detect_ner_with_regex_and_context(sentence, src_lang,  tag_type={'ID'}, prioritize_lang_match_over_ignore=True, ignore_stdnum_type={'isil', 'isbn', 'isan', 'imo', 'gs1_128', 'grid', 'figi', 'ean', 'casrn', 'cusip' }, all_regex=None, context_window=20, min_id_length=6, max_id_length=50,):
      """
      Output:
       - This function returns a list of 4 tuples, representing an NER detection for [(entity, start, end, tag), ...]
      Input:
       :sentence: the sentence to tag
       :src_lang: the language of the sentence
       :context_window: the contxt window in characters to check for context characters for any rules that requries context
       :max_id_length: the maximum length of an ID
       :tag_type: the type of NER tags we are detecting. If None, then detect everything.
       :ignore_stdnum_type: the set of stdnum we will consider NOT PII and not match as an ID
       :prioritize_lang_match_over_ignore: if true, and an ID matches an ingore list, we still keep it as an ID if there was an ID match for this particular src_lang
       :all_regex: a rulebase of the form {tag: {lang: [(regex, context, block), ...], 'default': [(regex, context, block), ...]}}. 
         context are words that must be found surronding the entity. block are words that must not be found.
         If all_regex is none, then we use the global regex_rulebase
       :do_context_check: if we require a context match
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
      def test_if_id_is_date(ent, tag, sentence, is_cjk):
        """
        Helper function used to test if an ent is a date or not
        We use dateparse to find context words around the ID/date to determine if its a date or not.
        """
        if len(ent) > 8: return ent, tag
        is_date =  dateparser.parse(ent, languages=[date_parser_lang_mapper.get(src_lang,src_lang)]) # use src_lang to make it faster, languages=[src_lang])
        if (is_date and tag == 'ID'):
            i = sentence.index(ent)
            len_ent = len(ent)
            j = i + len_ent
            #for speed we can just use these 6 windows to check for a date.
            #but for completeness we could check a sliding window. 
            #Maybe in some countries a year could
            #be in the middle of a date: Month Year Day
            ent_spans = [(-1,0), (-2, 0), (-3, 0), \
              (0, 1), (0, 2), (0, 3)]
            before = sentence[:i]
            after = sentence[j:]
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
                ent2 = "".join(before1)+" "+ent+" "+"".join(after1)
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
                  ent2 = "".join(before1)+" "+ent+" "+"".join(after1)
                ent = ent2.strip()
                tag = "DATE"
                break
        return ent, tag

      # main
      # if we are doing 'ID', we would still want to see if we catch DATE and ADDRESS. 
      # DATE and ADDRESS may have higher precedence, in which case it might overide an ID match. 
      sw = stopwords.get(src_lang, {})
      no_date = False
      if tag_type is not None and 'ID' in tag_type and 'DATE' not in tag_type:
         no_date = True
         tag_type = set(list(tag_type)+['DATE'])
      no_address = False
      if tag_type is not None and 'ID' in tag_type and 'ADDRESS' not in tag_type:
         no_address = True
         tag_type = set(list(tag_type)+['ADDRESS'])
        
      is_cjk = src_lang in ("zh", "ko", "ja")
      if is_cjk:
          sentence_set = set(sentence.lower())
      else:
          sentence_set = set(sentence.lower().split(" "))
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
          for regex_context in regex_group.get(src_lang, []) + regex_group.get("default", []):
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
                                is_stdnum = any(a for a in stnum_type if "." in a and country_2_lang.get(a.split(".")[0]) == src_lang)
                          if not ent_is_4_digit and not is_stdnum and any(a for a in stnum_type if a in ignore_stdnum_type):
                            #print ('ignoring this id because it is not pii', ent, stnum_type)
                            continue
                          #this is actually an ID and not a DATE
                          if any(a for a in stnum_type if a not in ignore_stdnum_type):
                            tag = 'ID'
                            is_stdnum = True
                            
                      #let's check the FIRST instance of this id is really a date; ideally we should do this for every instance of this ID
                      if tag == 'ID' and not is_stdnum:
                          ent, tag = test_if_id_is_date(ent, tag, sentence, is_cjk)
      
                      #now let's turn all occurances of ent in this sentence into a span mention and also check for context
                      len_ent = len(ent)
                      while True:
                        if ent not in sentence2:
                          break
                        else:
                          i = sentence2.index(ent)
                          j = i + len_ent
                          if potential_context or block:
                              len_sentence = len(sentence2)
                              left = sentence2[max(0, i - context_window) : i].lower()
                              right = sentence2[j : min(len_sentence, j + context_window)].lower()
                              found_context = False
                              if context:
                                for c in context:
                                  c = c.lower()
                                  if c in left or c in right:
                                      found_context = True
                                      break
                                
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
                            all_ner.append((ent, delta+i, delta+j, tag))
                          sentence2 = sentence2[i+len(ent):]
                          delta += j
                            
      all_ner = list(set(all_ner))
      # sort by length and position, favoring non-IDs first.
      # this doesn't do a perfect overlap match; just an overlap to the prior item.
      all_ner.sort(key=lambda a: a[1]+(1.0/(1.0+((a[3]!='ID'))+a[2]-a[1])))
      if not tag_type or 'ID' in tag_type:
        # now do overlaps prefering longer ents, and dates and addresses over embedded IDs or dates
        all_ner2 = []
        prev_mention = None
        for mention in all_ner:
          if prev_mention:
            if prev_mention[3] in ('ID', 'DATE', 'ADDRESS') and prev_mention[2] >= mention[1] and prev_mention[2] >= mention[2]:
              # if there is a complete overlap to an ID in an ADDRESS or a DATE, we ignore this ID
              # this is because we have more context for the DATE or ADDRESS to determine it is so. 
              if mention[3] in ('DATE', 'ID', 'PHONE'): 
                continue
            else:
              prev_mention = mention
          else:
            prev_mention = mention
          all_ner2.append(mention)
        all_ner = all_ner2
      if no_date:
         all_ner = [a for a in all_ner if a[3] != 'DATE']
      if no_address:
         all_ner = [a for a in all_ner if a[3] != 'ADDRESS']
     
      return all_ner
