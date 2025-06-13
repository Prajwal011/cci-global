```mermaid
flowchart LR
  Signup[📄 Signup Page]
  Signin[🔐 Signin Page]
  Dashboard[📊 Dashboard]

  Signup --> Signin
  Signin --> Dashboard

  subgraph Signup Page
    A1[Logo + Header]
    A2[Form: First / Last Name, Email, Password]
    A3[Sign Up Button / Social Options]
  end

  subgraph Signin Page
    B1[Logo + Header]
    B2[Social Login Buttons]
    B3[Form: Email + Password]
    B4[Sign In Button + Forgot Link]
  end

  subgraph Dashboard Page
    C1[Sidebar: Navigation Menu]
    C2[Topbar: Profile, Search, Notifications]
    C3[Main Panel: Widgets + Tables]
  end
