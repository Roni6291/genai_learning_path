# Smart City Crisis Simulation - Prompt Engineering Exercise

## Scenario Overview
**Location:** Novapolis (Fictional Smart City)  
**Crisis:** Major Power Outage  
**Role:** Crisis Response Assistant Consultant  
**Objective:** Use prompt engineering to guide AI through crisis resolution

---

## Challenge 1: Root Cause Analysis

### Engineered Prompt

```
You are a Senior Technical Investigator specializing in smart city infrastructure and power grid systems. You have 15+ years of experience analyzing critical infrastructure failures and possess expertise in electrical engineering, cybersecurity, and emergency management.

CONTEXT:
Novapolis is a smart city with a population of 2.5 million people. The city operates on an integrated power grid that supports:
- Residential zones (40% load)
- Commercial districts (30% load)
- Industrial sectors (20% load)
- Critical infrastructure including hospitals, data centers, and emergency services (10% load)

The city uses IoT sensors throughout the grid, automated distribution systems, and has implemented smart meters in 95% of locations.

INCIDENT:
At 14:37 local time, a complete citywide blackout occurred. All grid sectors went offline simultaneously. Backup generators at critical facilities activated, but the main grid remains non-functional. Initial reports indicate no immediate physical damage visible at major substations.

TASK:
Conduct a root cause analysis and identify exactly 3 plausible causes for this blackout. For each cause, provide:

1. **Cause Name**: A clear, specific title
2. **Likelihood Assessment**: Rate as High/Medium/Low with justification
3. **Technical Explanation**: Describe the failure mechanism (2-3 sentences)
4. **Supporting Evidence**: List 3 observable indicators that would confirm this cause
5. **Initial Response Actions**: Suggest 2-3 immediate investigation steps

Structure your analysis in a clear, prioritized format starting with the most likely cause.
```

## Challenge 2: Impact Assessment

### Engineered Prompt

```
You are the Chief City Planner for Novapolis with 20 years of experience in urban infrastructure management and emergency preparedness. You specialize in interdependency analysis of city systems and have managed responses to multiple infrastructure crises.

CURRENT SITUATION:
The citywide power outage that began at 14:37 is now entering its third hour (17:45). While backup generators are supporting some critical facilities, most of the city remains without power. Temperature is 32°C (89°F), and sunset is at 19:20.

CITY SECTORS:
1. **Healthcare Sector**: 8 hospitals, 45 clinics, 12 elderly care facilities
2. **Transportation**: Metro system (500K daily riders), traffic signals (2,000+ intersections), electric bus fleet
3. **Business/Commercial**: Downtown financial district, 15 shopping centers, 200+ restaurants, tech parks
4. **Residential**: 850K households across diverse income levels, high-rise apartments, suburban areas
5. **Other**: Water treatment plants, emergency services, telecommunications hubs

YOUR TASK:
Conduct a comprehensive impact assessment of this power outage. For each of the four main sectors (Healthcare, Transportation, Business/Commercial, Residential):

1. **Risk Level**: Classify as CRITICAL, HIGH, MODERATE, or LOW
2. **Immediate Impacts** (0-6 hours): Describe the most urgent consequences
3. **Cascading Effects** (6-24 hours): Explain how impacts will worsen or spread
4. **Vulnerable Populations**: Identify who is most at risk and why
5. **Time-Sensitivity Score** (1-10): Rate urgency where 10 = life-threatening within hours

Rank the sectors from highest to lowest priority for emergency response resources.

DELIVERABLE FORMAT:
Present as a prioritized action matrix that emergency managers can use for decision-making.
```

## Challenge 3: Crisis Communication

### Engineered Prompt

```
You are the Director of Public Communications for Novapolis with expertise in crisis messaging and public relations. You have successfully managed communications during previous emergencies and understand the principles of clear, calm, and actionable public messaging.

CRISIS CONTEXT:
- Citywide power outage began 3 hours ago (14:37)
- Cause still under investigation
- Estimated restoration time: 12-18 hours
- Backup power active at hospitals and emergency services
- No casualties reported so far
- Temperature is high (32°C/89°F), evening approaching

AUDIENCE DEMOGRAPHICS:
- 2.5 million residents with diverse backgrounds
- 15% elderly population (65+)
- 30% non-native speakers
- Mix of urban apartment dwellers and suburban homeowners
- High smartphone penetration but limited power for charging

YOUR TASK:
Write a public safety announcement that will be broadcast via:
- Emergency text alerts (160 characters)
- Radio broadcast (2-minute message)
- Social media (Twitter/X post, ~280 characters)
- Full press release (for news outlets)

REQUIREMENTS FOR EACH MESSAGE:

**Emergency Text Alert:**
- Maximum 160 characters
- Most critical information only
- Clear call to action

**Radio Broadcast:**
- 2 minutes when read aloud (approximately 250-300 words)
- Opening: Acknowledge the situation calmly
- Middle: Provide essential safety guidance (3-5 key points)
- Closing: Reassure and provide next update time
- Use simple language (8th-grade reading level)
- Avoid technical jargon
- Include specific actions citizens should take NOW

**Social Media Post:**
- ~280 characters for Twitter/X
- Shareable and clear
- Include relevant hashtag
- Link to more information

**Press Release:**
- Professional tone for media outlets
- Include: situation summary, current actions, public guidance, next update schedule
- Quote from city official
- Contact information for media inquiries

TONE GUIDELINES:
✓ Calm and authoritative (not alarming)
✓ Empathetic to citizen concerns
✓ Action-oriented with clear steps
✓ Transparent about what is known/unknown
✓ Reassuring but realistic

Avoid:
✗ Panic-inducing language
✗ Over-promising quick fixes
✗ Minimizing legitimate concerns
✗ Complex technical explanations


Some examples for Output Quality Standards

**Good Example** (Radio broadcast opening):
> "Novapolis residents, this is an official message from your city emergency management. We are currently experiencing a citywide power outage that began this afternoon at 2:37 PM. Our teams are working to restore power safely..."

**Poor Example** (avoid):
> "Don't panic! We have a massive grid failure but everything is under control..."
```

## Challenge 4: Optimization Strategy

### Stage 1: Initial Allocation Prompt

```
You are a Senior Logistics and Operations Expert specializing in emergency resource allocation. You have a Ph.D. in Operations Research and 12 years of field experience optimizing disaster response. You're skilled at data-driven decision-making under constraints.

MISSION:
Allocate 50 portable emergency power generators across Novapolis's 5 districts to minimize disruption and harm during the ongoing blackout.

DISTRICT PROFILES:

**District 1 - Central Business District**
- Population: 50,000 (daytime: 200,000)
- Key facilities: Main hospital (600 beds), financial district, city hall, central police station
- Current status: Complete blackout, backup power at hospital only

**District 2 - North Residential**
- Population: 600,000
- Key facilities: 2 community hospitals (200 beds each), 15 schools, elderly care home (300 residents)
- Current status: Complete blackout, elderly care facility on limited backup (4 hours remaining)

**District 3 - Industrial Zone**
- Population: 80,000 (daytime: 150,000)
- Key facilities: Water treatment plant (serves 40% of city), food warehouses, manufacturing
- Current status: Blackout, water treatment on backup (6 hours remaining)

**District 4 - East Suburbs**
- Population: 900,000
- Key facilities: 3 hospitals (150 beds each), metro depot, telecommunication hub
- Current status: Complete blackout, hospitals on backup power (varying durations)

**District 5 - South Mixed-Use**
- Population: 400,000
- Key facilities: University (20,000 students), convention center (emergency shelter), fire stations
- Current status: Complete blackout, fire stations on backup

GENERATOR SPECIFICATIONS:
- Each generator can power: 1 critical facility OR 1 city block (≈100 homes) OR 1 transportation hub
- Fuel supply: 12 hours per generator (can be refueled)
- Deployment time: 30 minutes per generator

YOUR TASK:
Provide a generator allocation strategy. Present your allocation as:

**District [Number] - [Name]: [X] generators**
- Primary allocation: [Specific facilities]
- Rationale: [One sentence]

Total must equal exactly 50 generators.
```

### Stage 2: Justification and Trade-offs Prompt

```
Based on your allocation of 50 generators across the 5 districts, now provide a comprehensive justification that includes:

1. **Allocation Philosophy**: What overarching principle guided your decisions? (e.g., life-safety first, utilitarian approach, cascading impact prevention)

2. **Prioritization Framework**: Explain the criteria you used to rank needs. Assign weights if applicable (e.g., life-safety: 40%, infrastructure criticality: 30%, population served: 20%, economic impact: 10%)

3. **Trade-off Analysis**: For each district, explicitly identify:
   - What you chose to support
   - What you chose NOT to support
   - Why the supported option outweighed alternatives
   - What risks you accepted by not allocating more resources there

4. **Vulnerable Groups**: How did your allocation address the needs of:
   - Elderly and medically dependent residents
   - Low-income areas without backup resources
   - Essential workers who keep the city functioning

5. **Temporal Reasoning**: Explain if your allocation considers:
   - Time-to-critical-failure for various systems
   - Whether resources could be redistributed after initial stabilization
   - Fuel resupply logistics

6. **Alternative Scenarios**: Briefly describe how your allocation would change if:
   - You had 30 generators (resource scarcity)
   - You had 80 generators (resource abundance)
   - The outage extended to 48+ hours

7. **Unintended Consequences**: What negative outcomes might result from your allocation strategy?

Present this as a decision memo that would be reviewed by the city's emergency management director.
```

---

## Challenge 5: Post-Crisis Plan

### Engineered Prompt

```
You are the Chief Policy Advisor to the Mayor of Novapolis, specializing in urban resilience and critical infrastructure policy. You hold a Master's in Public Policy and have advised multiple cities on infrastructure modernization. You excel at translating technical requirements into implementable policy frameworks.

MISSION:
Develop a comprehensive Post-Crisis Prevention and Resilience Plan to ensure Novapolis never experiences a similar catastrophic blackout again.

CRISIS LESSONS:
Based on the recent blackout:
- Single point of failure caused citywide impact
- Limited distributed backup power capacity
- Insufficient real-time monitoring and early warning
- Unclear emergency communication protocols
- Resource allocation delays due to lack of pre-planning
- Vulnerable populations disproportionately affected

YOUR TASK:
Create a strategic policy framework with recommendations across five categories:

**1. INFRASTRUCTURE RESILIENCE (Technical & Engineering)**
Propose 3-5 concrete infrastructure improvements:
- Grid modernization measures
- Redundancy and failsafe systems
- Distributed energy resources (solar, battery storage)
- Physical security enhancements

For each recommendation:
- Description: What specifically should be built/implemented?
- Timeline: Short-term (1 year), Medium-term (1-3 years), Long-term (3-5 years)
- Estimated Cost: Rough budget category ($ = <$10M, $$ = $10-50M, $$$ = >$50M)
- Risk Reduction: What specific failure mode does this prevent?

**2. OPERATIONAL PREPAREDNESS (Planning & Response)**
Recommend operational improvements:
- Emergency response protocols
- Resource pre-positioning strategies
- Training and drill requirements
- Inter-agency coordination mechanisms

**3. TECHNOLOGY & MONITORING (Early Warning Systems)**
Suggest technology investments:
- Predictive monitoring systems
- AI/ML for grid anomaly detection
- IoT sensor deployment
- Real-time dashboard for decision-makers

**4. POLICY & GOVERNANCE (Regulatory Framework)**
Propose policy changes:
- Building code updates (backup power requirements)
- Critical facility designation and standards
- Private sector resilience requirements
- Mutual aid agreements with neighboring cities
- Insurance and liability framework

**5. COMMUNITY RESILIENCE (Social & Equity Dimensions)**
Address societal vulnerabilities:
- Support programs for vulnerable populations
- Community energy hubs in each district
- Public education and preparedness campaigns
- Equity considerations in infrastructure investment

**6. FUNDING & IMPLEMENTATION ROADMAP**
- Identify funding sources (municipal budget, state/federal grants, public-private partnerships)
- Prioritize recommendations by impact and feasibility
- Define success metrics and accountability measures
- Establish governance structure for oversight

DELIVERABLE FORMAT:
Structure as a formal policy recommendation document with:
- Executive Summary (key recommendations)
- Detailed recommendations by category
- Implementation roadmap (quick wins vs. long-term investments)
- Risk assessment if recommendations are NOT implemented

CONSTRAINTS TO CONSIDER:
- City annual budget: $2.5 billion (infrastructure is 15%)
- Political feasibility (must balance costs with public acceptance)
- Technical feasibility (existing infrastructure is 30+ years old)
- Timeline (mayor faces re-election in 18 months)
```
