--
-- PostgreSQL database cluster dump
--

SET default_transaction_read_only = off;

SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;

--
-- Drop databases (except postgres and template1)
--





--
-- Drop roles
--

DROP ROLE postgres;


--
-- Roles
--

CREATE ROLE postgres;
ALTER ROLE postgres WITH SUPERUSER INHERIT CREATEROLE CREATEDB LOGIN REPLICATION BYPASSRLS PASSWORD 'SCRAM-SHA-256$4096:Cy/pYZ8iRTdnCo9+Er97Eg==$CO4abq2vcV1hMrPVrTEjnEKFfYcrilQ3QiN5nvMm4L8=:awk0RU1LvoI5aLMwreClBdJy7/YBOsV53R8AM2HU3hw=';

--
-- User Configurations
--








--
-- Databases
--

--
-- Database "template1" dump
--

--
-- PostgreSQL database dump
--

-- Dumped from database version 16.4 (Debian 16.4-1.pgdg120+1)
-- Dumped by pg_dump version 16.4 (Debian 16.4-1.pgdg120+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

UPDATE pg_catalog.pg_database SET datistemplate = false WHERE datname = 'template1';
DROP DATABASE template1;
--
-- Name: template1; Type: DATABASE; Schema: -; Owner: postgres
--

CREATE DATABASE template1 WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE_PROVIDER = libc LOCALE = 'en_US.utf8';


ALTER DATABASE template1 OWNER TO postgres;

\connect template1

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: DATABASE template1; Type: COMMENT; Schema: -; Owner: postgres
--

COMMENT ON DATABASE template1 IS 'default template for new databases';


--
-- Name: template1; Type: DATABASE PROPERTIES; Schema: -; Owner: postgres
--

ALTER DATABASE template1 IS_TEMPLATE = true;


\connect template1

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: DATABASE template1; Type: ACL; Schema: -; Owner: postgres
--

REVOKE CONNECT,TEMPORARY ON DATABASE template1 FROM PUBLIC;
GRANT CONNECT ON DATABASE template1 TO PUBLIC;


--
-- PostgreSQL database dump complete
--

--
-- Database "postgres" dump
--

--
-- PostgreSQL database dump
--

-- Dumped from database version 16.4 (Debian 16.4-1.pgdg120+1)
-- Dumped by pg_dump version 16.4 (Debian 16.4-1.pgdg120+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

DROP DATABASE postgres;
--
-- Name: postgres; Type: DATABASE; Schema: -; Owner: postgres
--

CREATE DATABASE postgres WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE_PROVIDER = libc LOCALE = 'en_US.utf8';


ALTER DATABASE postgres OWNER TO postgres;

\connect postgres

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: DATABASE postgres; Type: COMMENT; Schema: -; Owner: postgres
--

COMMENT ON DATABASE postgres IS 'default administrative connection database';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: ab_testing_results; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ab_testing_results (
    test_id integer NOT NULL,
    test_name character varying(255) NOT NULL,
    model_a_id integer,
    model_b_id integer,
    metric_name character varying(255) NOT NULL,
    model_a_metric_value double precision NOT NULL,
    model_b_metric_value double precision NOT NULL,
    winning_model_id integer,
    "timestamp" timestamp with time zone DEFAULT now()
);


ALTER TABLE public.ab_testing_results OWNER TO postgres;

--
-- Name: ab_testing_results_test_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.ab_testing_results_test_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.ab_testing_results_test_id_seq OWNER TO postgres;

--
-- Name: ab_testing_results_test_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.ab_testing_results_test_id_seq OWNED BY public.ab_testing_results.test_id;


--
-- Name: api_request_logs; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.api_request_logs (
    request_id integer NOT NULL,
    endpoint character varying(255) NOT NULL,
    request_method character varying(10) NOT NULL,
    request_body text,
    response_status integer NOT NULL,
    response_time_ms double precision,
    user_id integer,
    ip_address character varying(45),
    "timestamp" timestamp with time zone DEFAULT now()
);


ALTER TABLE public.api_request_logs OWNER TO postgres;

--
-- Name: api_request_logs_request_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.api_request_logs_request_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.api_request_logs_request_id_seq OWNER TO postgres;

--
-- Name: api_request_logs_request_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.api_request_logs_request_id_seq OWNED BY public.api_request_logs.request_id;


--
-- Name: error_logs; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.error_logs (
    error_id integer NOT NULL,
    error_type character varying(255),
    error_message text NOT NULL,
    model_id integer,
    user_id integer,
    "timestamp" timestamp with time zone DEFAULT now()
);


ALTER TABLE public.error_logs OWNER TO postgres;

--
-- Name: error_logs_error_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.error_logs_error_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.error_logs_error_id_seq OWNER TO postgres;

--
-- Name: error_logs_error_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.error_logs_error_id_seq OWNED BY public.error_logs.error_id;


--
-- Name: model_metadata; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.model_metadata (
    model_id integer NOT NULL,
    model_name character varying,
    version character varying,
    training_data text,
    training_start_time timestamp with time zone,
    training_end_time timestamp with time zone,
    accuracy double precision NOT NULL,
    f1_score double precision,
    "precision" double precision,
    recall double precision,
    training_loss double precision NOT NULL,
    validation_loss double precision NOT NULL,
    training_accuracy double precision,
    validation_accuracy double precision,
    training_params json NOT NULL,
    logs text,
    created_at timestamp without time zone,
    updated_at timestamp without time zone
);


ALTER TABLE public.model_metadata OWNER TO postgres;

--
-- Name: model_metadata_model_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.model_metadata_model_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.model_metadata_model_id_seq OWNER TO postgres;

--
-- Name: model_metadata_model_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.model_metadata_model_id_seq OWNED BY public.model_metadata.model_id;


--
-- Name: predictions; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.predictions (
    prediction_id integer NOT NULL,
    user_id integer,
    model_id integer,
    image_path character varying,
    prediction json,
    top_5_prediction json,
    confidence double precision,
    feedback_given boolean,
    feedback_comment text,
    predicted_at timestamp without time zone
);


ALTER TABLE public.predictions OWNER TO postgres;

--
-- Name: predictions_prediction_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.predictions_prediction_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.predictions_prediction_id_seq OWNER TO postgres;

--
-- Name: predictions_prediction_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.predictions_prediction_id_seq OWNED BY public.predictions.prediction_id;


--
-- Name: roles; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.roles (
    role_id integer NOT NULL,
    role_name character varying(50),
    role_description character varying(255)
);


ALTER TABLE public.roles OWNER TO postgres;

--
-- Name: roles_role_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.roles_role_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.roles_role_id_seq OWNER TO postgres;

--
-- Name: roles_role_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.roles_role_id_seq OWNED BY public.roles.role_id;


--
-- Name: users; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.users (
    user_id integer NOT NULL,
    username character varying(255) NOT NULL,
    hashed_password character varying(255) NOT NULL,
    email character varying(255),
    role_id integer NOT NULL,
    disabled boolean,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp without time zone
);


ALTER TABLE public.users OWNER TO postgres;

--
-- Name: users_user_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.users_user_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.users_user_id_seq OWNER TO postgres;

--
-- Name: users_user_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.users_user_id_seq OWNED BY public.users.user_id;


--
-- Name: ab_testing_results test_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ab_testing_results ALTER COLUMN test_id SET DEFAULT nextval('public.ab_testing_results_test_id_seq'::regclass);


--
-- Name: api_request_logs request_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.api_request_logs ALTER COLUMN request_id SET DEFAULT nextval('public.api_request_logs_request_id_seq'::regclass);


--
-- Name: error_logs error_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.error_logs ALTER COLUMN error_id SET DEFAULT nextval('public.error_logs_error_id_seq'::regclass);


--
-- Name: model_metadata model_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.model_metadata ALTER COLUMN model_id SET DEFAULT nextval('public.model_metadata_model_id_seq'::regclass);


--
-- Name: predictions prediction_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.predictions ALTER COLUMN prediction_id SET DEFAULT nextval('public.predictions_prediction_id_seq'::regclass);


--
-- Name: roles role_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.roles ALTER COLUMN role_id SET DEFAULT nextval('public.roles_role_id_seq'::regclass);


--
-- Name: users user_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users ALTER COLUMN user_id SET DEFAULT nextval('public.users_user_id_seq'::regclass);


--
-- Data for Name: ab_testing_results; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.ab_testing_results (test_id, test_name, model_a_id, model_b_id, metric_name, model_a_metric_value, model_b_metric_value, winning_model_id, "timestamp") FROM stdin;
\.


--
-- Data for Name: api_request_logs; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.api_request_logs (request_id, endpoint, request_method, request_body, response_status, response_time_ms, user_id, ip_address, "timestamp") FROM stdin;
\.


--
-- Data for Name: error_logs; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.error_logs (error_id, error_type, error_message, model_id, user_id, "timestamp") FROM stdin;
\.


--
-- Data for Name: model_metadata; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.model_metadata (model_id, model_name, version, training_data, training_start_time, training_end_time, accuracy, f1_score, "precision", recall, training_loss, validation_loss, training_accuracy, validation_accuracy, training_params, logs, created_at, updated_at) FROM stdin;
1	first model	v1	some data collected	2024-08-15 17:17:41.173+00	2024-08-15 17:17:41.173+00	0.9	0.89	0.45	0.9	0.9	0.9	0.78	0.67	{"my_pram": 0.009}	some random log	2024-08-15 17:20:05.062057	2024-08-15 17:20:05.062057
\.


--
-- Data for Name: predictions; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.predictions (prediction_id, user_id, model_id, image_path, prediction, top_5_prediction, confidence, feedback_given, feedback_comment, predicted_at) FROM stdin;
35	1	1	image path goes here	{"predicted_class": "cat"}	[{"class_label": "cat", "confidence_score": 0.85}, {"class_label": "dog", "confidence_score": 0.1}, {"class_label": "rabbit", "confidence_score": 0.03}, {"class_label": "hamster", "confidence_score": 0.01}, {"class_label": "parrot", "confidence_score": 0.01}]	0.9	f	feedback given by endpoints or edge devices	2024-08-15 17:22:11.915459
68	1	1	image path goes here	{"predicted_class": "Apple___healthy"}	[{"class_name": "Apple___healthy", "confidence": 0.9999995231628418}, {"class_name": "Background_without_leaves", "confidence": 4.373388264866662e-07}, {"class_name": "Cherry___Powdery_mildew", "confidence": 5.558124627214056e-08}, {"class_name": "Apple___Black_rot", "confidence": 1.8920200517413832e-08}, {"class_name": "Peach___healthy", "confidence": 8.438733267723819e-09}]	0.9999995231628418	f	feedback given by endpoints or edge devices	2024-08-15 17:25:19.527972
\.


--
-- Data for Name: roles; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.roles (role_id, role_name, role_description) FROM stdin;
1	admin	Do the administrative activities
38	farmer	farmer can only predict and view his own data
\.


--
-- Data for Name: users; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.users (user_id, username, hashed_password, email, role_id, disabled, created_at, updated_at) FROM stdin;
1	arif	$2b$12$t6Bguxsn.pvPshXcSb8P2OLESENy33WGSqBnAMoBQ1vdncwqV1fU6	arif@example.com	1	f	2024-08-14 22:22:12.826816+00	2024-08-14 22:22:12.826816
\.


--
-- Name: ab_testing_results_test_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.ab_testing_results_test_id_seq', 1, false);


--
-- Name: api_request_logs_request_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.api_request_logs_request_id_seq', 1, false);


--
-- Name: error_logs_error_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.error_logs_error_id_seq', 1, false);


--
-- Name: model_metadata_model_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.model_metadata_model_id_seq', 1, true);


--
-- Name: predictions_prediction_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.predictions_prediction_id_seq', 68, true);


--
-- Name: roles_role_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.roles_role_id_seq', 38, true);


--
-- Name: users_user_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.users_user_id_seq', 1, true);


--
-- Name: ab_testing_results ab_testing_results_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ab_testing_results
    ADD CONSTRAINT ab_testing_results_pkey PRIMARY KEY (test_id);


--
-- Name: api_request_logs api_request_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.api_request_logs
    ADD CONSTRAINT api_request_logs_pkey PRIMARY KEY (request_id);


--
-- Name: error_logs error_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.error_logs
    ADD CONSTRAINT error_logs_pkey PRIMARY KEY (error_id);


--
-- Name: model_metadata model_metadata_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.model_metadata
    ADD CONSTRAINT model_metadata_pkey PRIMARY KEY (model_id);


--
-- Name: predictions predictions_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.predictions
    ADD CONSTRAINT predictions_pkey PRIMARY KEY (prediction_id);


--
-- Name: roles roles_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.roles
    ADD CONSTRAINT roles_pkey PRIMARY KEY (role_id);


--
-- Name: users users_email_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_email_key UNIQUE (email);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (user_id);


--
-- Name: users users_username_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_username_key UNIQUE (username);


--
-- Name: ix_ab_testing_results_test_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_ab_testing_results_test_id ON public.ab_testing_results USING btree (test_id);


--
-- Name: ix_api_request_logs_request_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_api_request_logs_request_id ON public.api_request_logs USING btree (request_id);


--
-- Name: ix_error_logs_error_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_error_logs_error_id ON public.error_logs USING btree (error_id);


--
-- Name: ix_model_metadata_model_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_model_metadata_model_id ON public.model_metadata USING btree (model_id);


--
-- Name: ix_predictions_prediction_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_predictions_prediction_id ON public.predictions USING btree (prediction_id);


--
-- Name: ix_users_user_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_users_user_id ON public.users USING btree (user_id);


--
-- Name: ab_testing_results ab_testing_results_model_a_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ab_testing_results
    ADD CONSTRAINT ab_testing_results_model_a_id_fkey FOREIGN KEY (model_a_id) REFERENCES public.model_metadata(model_id);


--
-- Name: ab_testing_results ab_testing_results_model_b_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ab_testing_results
    ADD CONSTRAINT ab_testing_results_model_b_id_fkey FOREIGN KEY (model_b_id) REFERENCES public.model_metadata(model_id);


--
-- Name: ab_testing_results ab_testing_results_winning_model_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ab_testing_results
    ADD CONSTRAINT ab_testing_results_winning_model_id_fkey FOREIGN KEY (winning_model_id) REFERENCES public.model_metadata(model_id);


--
-- Name: api_request_logs api_request_logs_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.api_request_logs
    ADD CONSTRAINT api_request_logs_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id);


--
-- Name: error_logs error_logs_model_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.error_logs
    ADD CONSTRAINT error_logs_model_id_fkey FOREIGN KEY (model_id) REFERENCES public.model_metadata(model_id);


--
-- Name: error_logs error_logs_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.error_logs
    ADD CONSTRAINT error_logs_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id);


--
-- Name: predictions predictions_model_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.predictions
    ADD CONSTRAINT predictions_model_id_fkey FOREIGN KEY (model_id) REFERENCES public.model_metadata(model_id);


--
-- Name: predictions predictions_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.predictions
    ADD CONSTRAINT predictions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id);


--
-- Name: users users_role_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_role_id_fkey FOREIGN KEY (role_id) REFERENCES public.roles(role_id);


--
-- PostgreSQL database dump complete
--

--
-- PostgreSQL database cluster dump complete
--

