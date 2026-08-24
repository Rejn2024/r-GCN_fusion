"""Materialise the SSN/PROV-O/GeoSPARQL view alongside legacy Neo4j types.

The legacy labels and relationships remain available to existing r-GCN code.
Ontology labels, IRIs, predicate IRIs, geometries, and relation aliases make the
same database exportable without treating report claims as canonical facts.
"""

from __future__ import annotations

ESM = "https://w3id.org/rgcn-fusion/esm#"
GEO = "http://www.opengis.net/ont/geosparql#"
PROV = "http://www.w3.org/ns/prov#"
SOSA = "http://www.w3.org/ns/sosa/"
DATA = "https://w3id.org/rgcn-fusion/resource/"


def materialize_ontology_view(driver, database: str | None = None) -> dict[str, int]:
    """Add standards-aligned labels and edges to an already populated graph."""

    statements = [
        # Canonical graph.
        """
        MATCH (n)
        WHERE n:AircraftFamily OR n:AircraftVariant OR n:Radar OR n:RadarMode OR n:Operator
        SET n:CanonicalEntity:ProvEntity,
            n.iri = $data + replace(n.id, ':', '/'),
            n.ontology_iri = CASE
              WHEN n:AircraftFamily THEN $esm + 'AircraftFamily'
              WHEN n:AircraftVariant THEN $esm + 'AircraftVariant'
              WHEN n:RadarMode THEN $esm + 'RadarMode'
              WHEN n:Radar THEN $esm + 'Radar'
              ELSE $esm + 'Operator' END
        """,
        """MATCH (a:Operator)-[:OPERATES]->(b:AircraftVariant)
        MERGE (a)-[r:ESM_OPERATES]->(b) SET r.predicate_iri=$esm+'operates'""",
        """MATCH (a:AircraftVariant)-[:USES_RADAR]->(b:Radar)
        MERGE (a)-[r:ESM_USES_RADAR]->(b) SET r.predicate_iri=$esm+'usesRadar'""",
        """MATCH (a:Radar)-[:HAS_MODE]->(b:RadarMode)
        MERGE (a)-[r:ESM_HAS_MODE]->(b) SET r.predicate_iri=$esm+'hasMode'""",
        """MATCH (a:AircraftVariant)-[:VARIANT_OF]->(b:AircraftFamily)
        MERGE (a)-[r:ESM_VARIANT_OF]->(b) SET r.predicate_iri=$esm+'variantOf'""",
        # Evidence resources and the three DS masses.
        """
        MATCH (n:Observation)
        SET n:ESMObservation:ProvEntity, n.iri=$data+replace(n.id, ':', '/'),
            n.ontology_iri=$esm+'ESMObservation', n.result_time=n.timestamp_iso8601,
            n.mass_non_match=n.ds_masses[0], n.mass_match=n.ds_masses[1],
            n.mass_uncertain=n.ds_masses[2]
        """,
        """MATCH (n:CandidateEvidence)
        SET n:CandidateHypothesis:ProvEntity, n.iri=$data+replace(n.id, ':', '/'),
            n.ontology_iri=$esm+'CandidateHypothesis',
            n.mass_non_match=n.ds_masses[0], n.mass_match=n.ds_masses[1],
            n.mass_uncertain=n.ds_masses[2]""",
        """MATCH (n:Track)
        SET n:GeoFeature:ProvEntity, n.iri=$data+replace(n.id, ':', '/'),
            n.ontology_iri=$esm+'Track'""",
        """MATCH (n:IntelligenceReport)
        SET n:ProvEntity, n.iri=$data+replace(n.id, ':', '/'),
            n.ontology_iri=$esm+'IntelligenceReport'""",
        """MATCH (n:ReportClaim)
        SET n:Claim:ProvEntity, n.iri=$data+replace(n.id, ':', '/'),
            n.ontology_iri=$esm+'Claim', n.claim_object=n.object_id,
            n.mass_non_match=n.ds_masses[0], n.mass_match=n.ds_masses[1],
            n.mass_uncertain=n.ds_masses[2]""",
        # Standard relation aliases. Legacy edges intentionally remain in place.
        """MATCH (t:Track)-[:TRACK_HAS_OBSERVATION]->(o:Observation)
        MERGE (t)-[r:ESM_HAS_OBSERVATION]->(o) SET r.predicate_iri=$esm+'hasObservation'
        MERGE (o)-[q:ESM_IN_TRACK]->(t) SET q.predicate_iri=$esm+'inTrack'
        MERGE (o)-[f:SOSA_HAS_FEATURE_OF_INTEREST]->(t)
        SET f.predicate_iri=$sosa+'hasFeatureOfInterest'""",
        """MATCH (o:Observation)-[:HAS_CANDIDATE]->(c:CandidateEvidence)
        MERGE (o)-[r:ESM_HAS_CANDIDATE]->(c) SET r.predicate_iri=$esm+'hasCandidate'
        MERGE (c)-[q:PROV_WAS_DERIVED_FROM]->(o) SET q.predicate_iri=$prov+'wasDerivedFrom'""",
        """MATCH (r:IntelligenceReport)-[:REPORT_CONTAINS_CLAIM]->(c:ReportClaim)
        MERGE (r)-[a:ESM_CONTAINS_CLAIM]->(c) SET a.predicate_iri=$esm+'containsClaim'
        MERGE (c)-[b:PROV_WAS_DERIVED_FROM]->(r) SET b.predicate_iri=$prov+'wasDerivedFrom'""",
        """MATCH (c:ReportClaim)-[:CLAIM_SUPPORTS_CANDIDATE]->(h:CandidateEvidence)
        MERGE (c)-[r:ESM_SUPPORTS]->(h) SET r.predicate_iri=$esm+'supports'""",
        """MATCH (c:ReportClaim)-[:CLAIM_REFUTES_CANDIDATE]->(h:CandidateEvidence)
        MERGE (c)-[r:ESM_REFUTES]->(h) SET r.predicate_iri=$esm+'refutes'""",
        # Geometry nodes use CRS84, whose coordinate order is longitude latitude.
        """
        MATCH (o:Observation)
        WHERE o.estimated_longitude_deg IS NOT NULL AND o.estimated_latitude_deg IS NOT NULL
        MERGE (g:Geometry {id: o.id+':estimated-location'})
        SET g.iri=$data+replace(g.id, ':', '/'), g.ontology_iri=$geo+'Geometry',
            g.as_wkt='<http://www.opengis.net/def/crs/OGC/1.3/CRS84> POINT('+
                     toString(o.estimated_longitude_deg)+' '+toString(o.estimated_latitude_deg)+')',
            g.wkt_datatype=$geo+'wktLiteral'
        MERGE (o)-[r:ESM_ESTIMATED_LOCATION]->(g)
        SET r.predicate_iri=$esm+'estimatedLocation'
        """,
        """
        MATCH (o:Observation)
        MERGE (result:SignalMeasurement {id:o.id+':signal-result'})
        SET result:ProvEntity, result.iri=$data+replace(result.id, ':', '/'),
            result.ontology_iri=$esm+'SignalMeasurement',
            result.measured_centre_frequency_ghz=o.measured_centre_frequency_ghz,
            result.measured_bandwidth_mhz=o.measured_bandwidth_mhz,
            result.measured_pulse_width_us=o.measured_pulse_width_us,
            result.measured_prf_hz=o.measured_prf_hz,
            result.measured_scan_period_s=o.measured_scan_period_s,
            result.measured_dwell_time_ms=o.measured_dwell_time_ms,
            result.measured_duty_cycle=o.measured_duty_cycle,
            result.measured_erp_dbm=o.measured_erp_dbm
        MERGE (o)-[r:SOSA_HAS_RESULT]->(result)
        SET r.predicate_iri=$sosa+'hasResult'
        """,
        """
        MATCH (t:Track)
        CALL {
          WITH t
          MATCH (t)-[:TRACK_HAS_OBSERVATION]->(o:Observation)
          WHERE o.estimated_longitude_deg IS NOT NULL AND o.estimated_latitude_deg IS NOT NULL
          WITH o ORDER BY o.sequence_index
          RETURN collect(toString(o.estimated_longitude_deg)+' '+
                         toString(o.estimated_latitude_deg)) AS coordinates
        }
        WHERE size(coordinates) > 1
        MERGE (trajectory:Trajectory {id:t.id+':trajectory'})
        SET trajectory:GeoFeature:ProvEntity,
            trajectory.iri=$data+replace(trajectory.id, ':', '/'),
            trajectory.ontology_iri=$esm+'Trajectory'
        MERGE (geometry:Geometry {id:trajectory.id+':geometry'})
        SET geometry.iri=$data+replace(geometry.id, ':', '/'),
            geometry.ontology_iri=$geo+'Geometry',
            geometry.as_wkt='<http://www.opengis.net/def/crs/OGC/1.3/CRS84> LINESTRING('+
              reduce(wkt='', coordinate IN coordinates |
                wkt+CASE WHEN wkt='' THEN '' ELSE ',' END+coordinate)+')',
            geometry.wkt_datatype=$geo+'wktLiteral'
        MERGE (trajectory)-[g:GEO_HAS_GEOMETRY]->(geometry)
        SET g.predicate_iri=$geo+'hasGeometry'
        MERGE (t)-[r:ESM_HAS_TRAJECTORY]->(trajectory)
        SET r.predicate_iri=$esm+'hasTrajectory'
        """,
    ]
    with driver.session(**({"database": database} if database else {})) as session:
        for statement in statements:
            session.run(statement, esm=ESM, geo=GEO, prov=PROV, sosa=SOSA, data=DATA).consume()
        summary = session.run(
            """MATCH (n) WHERE n.ontology_iri IS NOT NULL
            RETURN count(n) AS resources"""
        ).single()
    return {"ontology_resources": int(summary["resources"])}
