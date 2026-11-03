# FarmVerse Nexus - Module 2: Farm Management

Spring Boot 3 + PostgreSQL. Backend (REST API) only. Covers: farms, plots, soil tests, land documents, worker assignment, farm summary.

## Run
1. Install Java 17+ and PostgreSQL (Maven is built into IntelliJ).
2. Create the database once (pgAdmin or psql): `CREATE DATABASE farmverse_nexus;`
3. Set your PostgreSQL password: either edit `src/main/resources/application.properties` or set the env vars `DB_URL`, `DB_USERNAME`, `DB_PASSWORD`, `UPLOAD_DIR`.
4. Run `FarmManagementApplication` from IntelliJ (green play button) -> http://localhost:8080
Tables are created automatically.

## Endpoints
| Method | URL | Purpose |
|---|---|---|
| POST | /api/farms | Add farm |
| GET | /api/farms[?ownerId=\|?workerId=] | List farms (all / owned by a user / assigned to a field worker) |
| GET/PUT/DELETE | /api/farms/{id} | View / edit / delete farm (PUT body has no `ownerId`) |
| GET | /api/farms/{id}/summary | Farm summary page |
| POST/GET | /api/farms/{farmId}/plots | Add / list plots |
| GET/PUT/DELETE | /api/plots/{id} | View / edit / delete plot |
| POST/GET | /api/plots/{plotId}/soil-tests | Add / list soil tests |
| GET/PUT/DELETE | /api/soil-tests/{id} | View / edit / delete a soil test |
| POST | /api/farms/{farmId}/workers/{userId} | Assign worker |
| DELETE | /api/farms/{farmId}/workers/{userId} | Unassign worker |
| GET | /api/farms/{farmId}/workers | List assigned workers |
| POST | /api/farms/{farmId}/documents | Upload (form-data: `file`, `docType`) |
| GET | /api/farms/{farmId}/documents | List documents |
| GET | /api/documents/{id}/download | Download document |
| DELETE | /api/documents/{id} | Delete document |

## Sample requests (Postman, body = raw JSON)
POST /api/farms
{ "name": "Green Valley", "ownerId": 1, "village": "Anandapuram", "district": "Visakhapatnam",
  "state": "Andhra Pradesh", "latitude": 17.9, "longitude": 83.4, "totalAreaAcres": 10 }

POST /api/farms/1/plots
{ "name": "North Field", "areaAcres": 4, "soilType": "red" }

POST /api/plots/1/soil-tests
{ "testDate": "2026-10-20", "ph": 6.5, "nitrogen": 120, "phosphorus": 40, "potassium": 90, "notes": "Good" }

## Rules built in
- Sum of plot areas can never exceed the farm's total area.
- Farm area cannot be reduced below the area already divided into plots.
- Same worker cannot be assigned twice to a farm.
- Only PDF / JPG / PNG uploads, max 10MB. The file's real signature is checked, not just its extension.
- Plot names are unique within a farm; a farm name is unique per owner (case-insensitive).
- A field worker can list their farms with `GET /api/farms?workerId=`.
- Farm summary includes each plot's latest soil test and the assigned workers.
- Deleting a farm also deletes its plots, soil tests, workers and documents.
- Clean JSON errors (400 / 404) from GlobalExceptionHandler.

## For the team (what other modules use from you)
`farm.id`, `farm.latitude/longitude` (Weather), `plot.id`, `plot.areaAcres`, `plot.soilType` (GIS, Crops, Irrigation).

## Integration with Module 1 (the only work left)
Module 2 talks to Module 1 through one interface: `integration/UserDirectory`
(`isFieldWorker(userId)`, `displayName(userId)`).
1. Implement `UserDirectory` using Module 1's user service.
2. Delete `integration/StubUserDirectory.java` (it accepts every user id for now).
   Worker assignment validation and worker names in the farm summary then work automatically.
3. Add role / ownership checks in the controllers (Admin: all farms; Farmer: own farms only;
   Field Worker: read-only, assigned farms only). This needs Module 1's login/JWT, so it is not done yet.

## Tests
`mvn test` runs unit tests for plot area/duplicate rules, worker assignment, farm update/list rules and upload validation.
